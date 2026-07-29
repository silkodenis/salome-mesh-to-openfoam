#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct SALOME SMESH -> OpenFOAM polyMesh exporter (Python 3).

Designed for SALOME 9.x. The exporter reads the selected SMESH mesh directly
and writes OpenFOAM's points/faces/owner/neighbour/boundary/cellZones files.
No UNV, MED, Gmsh, or remeshing is involved. Since cells are exported through
their faces, tetrahedra, prisms, pyramids, hexahedra and general polyhedra are
handled uniformly.

Usage inside SALOME:
  1. Select exactly one completed mesh in the Object Browser.
  2. File -> Load Script (Ctrl+T), select this file.
  3. Select the root directory of an OpenFOAM case in the dialog.
  4. Run checkMesh in that case after export.

If the directory dialog is unavailable, set OUTPUT_CASE_DIR below.

Face groups become OpenFOAM boundary patches.
Volume groups become OpenFOAM cellZones.

This is an independent Python 3 implementation inspired by the direct-export
approach of Nicolas Edh's GPL-3.0 salomeToOpenFOAM project:
https://github.com/nicolasedh/salomeToOpenFOAM

SPDX-License-Identifier: GPL-3.0-or-later
"""

from array import array
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
import re
import struct
import tempfile
import time


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Leave empty to choose the OpenFOAM case directory in a dialog.
OUTPUT_CASE_DIR = ""

# Exact SALOME face-group name -> OpenFOAM patch type.
# Other groups containing "wall" in their name are also written as wall.
PATCH_TYPES = {
    "airplane": "wall",
    "airframe": "wall",
    "stabilizer": "wall",
    "canard1": "wall",
    "canard2": "wall",
    "canard3": "wall",
    "canard4": "wall",
}

DEFAULT_PATCH_TYPE = "patch"
DEFAULT_PATCH_NAME = "defaultFaces"

# Validate and, if necessary, reverse the node order of every newly found face.
# Recommended. It costs CPU time but avoids inward-oriented owner faces.
VERIFY_FACE_ORIENTATION = True

# Print progress after this many volume cells.
PROGRESS_INTERVAL = 25000


FOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
| Direct export from SALOME SMESH to OpenFOAM                                |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {foam_class};
    location    \"constant/polyMesh\";
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def log(message):
    print(f"[salomeToOpenFOAM] {message}", flush=True)


def foam_word(name, fallback):
    """Return a valid, reasonably readable OpenFOAM word."""
    value = re.sub(r"[^A-Za-z0-9_+.-]+", "_", str(name).strip())
    value = value.strip("_")
    if not value:
        value = fallback
    if value[0].isdigit():
        value = "group_" + value
    return value


def unique_name(requested, used):
    name = requested
    suffix = 2
    while name in used:
        name = f"{requested}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def vector_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def polygon_area_vector(face_point_ids, points):
    """Newell-like polygon area vector (twice the signed area vector)."""
    ax = ay = az = 0.0
    count = len(face_point_ids)
    for i, point_id in enumerate(face_point_ids):
        p = points[point_id]
        q = points[face_point_ids[(i + 1) % count]]
        ax += p[1] * q[2] - p[2] * q[1]
        ay += p[2] * q[0] - p[0] * q[2]
        az += p[0] * q[1] - p[1] * q[0]
    return (ax, ay, az)


def average_point(point_ids, points):
    count = len(point_ids)
    sx = sy = sz = 0.0
    for point_id in point_ids:
        p = points[point_id]
        sx += p[0]
        sy += p[1]
        sz += p[2]
    return (sx / count, sy / count, sz / count)


def orient_face_outward(face_point_ids, cell_point_ids, points):
    """Orient a face so its area vector points away from its owner cell."""
    if len(face_point_ids) < 3:
        raise RuntimeError(f"Face with fewer than 3 points: {face_point_ids}")
    face = tuple(face_point_ids)
    face_centre = average_point(face, points)
    cell_centre = average_point(cell_point_ids, points)
    outward_direction = vector_sub(face_centre, cell_centre)
    area_vector = polygon_area_vector(face, points)
    if vector_dot(area_vector, outward_direction) < 0.0:
        face = tuple(reversed(face))
    return face


def face_key(point_ids):
    """Topology key independent of face orientation."""
    ordered = sorted(point_ids)
    return struct.pack(f"<{len(ordered)}q", *ordered)


def get_selected_mesh():
    import salome
    import SMESH
    from salome.smesh import smeshBuilder

    count = salome.sg.SelectedCount()
    if count != 1:
        raise RuntimeError(
            f"Select exactly one completed mesh in SALOME (selected: {count})."
        )

    entry = salome.sg.getSelected(0)
    study_object = salome.myStudy.FindObjectID(entry)
    if study_object is None:
        raise RuntimeError("SALOME could not resolve the selected study object.")

    selected_object = study_object.GetObject()
    smesh = smeshBuilder.New(salome.myStudy)
    mesh = smesh.Mesh(selected_object)

    if mesh.NbVolumes() <= 0:
        raise RuntimeError(
            "The selected object has no volume cells. Select the complete 3D mesh, "
            "not a group or sub-mesh."
        )

    return mesh, SMESH, study_object.GetName()


def choose_case_directory():
    if OUTPUT_CASE_DIR:
        return Path(OUTPUT_CASE_DIR).expanduser().resolve()

    try:
        from PyQt5.QtWidgets import QFileDialog

        selected = QFileDialog.getExistingDirectory(
            None,
            "Select OpenFOAM case directory",
            str(Path.home()),
        )
        if not selected:
            raise RuntimeError("Export cancelled: no OpenFOAM case directory selected.")
        return Path(selected).resolve()
    except ImportError as exc:
        raise RuntimeError(
            "Directory dialog is unavailable. Edit OUTPUT_CASE_DIR at the top of "
            "the script and run it again."
        ) from exc


def collect_group_data(mesh, SMESH, node_map):
    """Collect patch face membership and volume groups before cell traversal."""
    patch_names = []
    patch_types = []
    patch_by_key = {}
    volume_groups = []
    used_names = set()

    for group in mesh.GetGroups():
        group_type = group.GetType()
        raw_name = group.GetName()

        if group_type == SMESH.FACE:
            patch_name = unique_name(
                foam_word(raw_name, "patch"), used_names
            )
            patch_id = len(patch_names)
            patch_names.append(patch_name)
            requested_type = PATCH_TYPES.get(raw_name)
            if requested_type is None and "wall" in raw_name.lower():
                requested_type = "wall"
            patch_types.append(requested_type or DEFAULT_PATCH_TYPE)

            for salome_face_id in group.GetIDs():
                salome_nodes = mesh.GetElemNodes(salome_face_id)
                try:
                    point_ids = tuple(node_map[node_id] for node_id in salome_nodes)
                except KeyError as exc:
                    raise RuntimeError(
                        f"Face group '{raw_name}' references an unknown node: {exc}"
                    ) from exc
                key = face_key(point_ids)
                previous = patch_by_key.get(key)
                if previous is not None and previous != patch_id:
                    raise RuntimeError(
                        f"A boundary face belongs to both '{patch_names[previous]}' "
                        f"and '{patch_name}'. Boundary face groups must not overlap."
                    )
                patch_by_key[key] = patch_id

        elif group_type == SMESH.VOLUME:
            zone_name = unique_name(
                foam_word(raw_name, "cellZone"), used_names
            )
            volume_groups.append((zone_name, list(group.GetIDs())))

    return patch_names, patch_types, patch_by_key, volume_groups


def iter_cell_faces(mesh, volume_id):
    face_index = 0
    while True:
        nodes = mesh.GetElemFaceNodes(volume_id, face_index)
        if not nodes:
            break
        yield nodes
        face_index += 1


def make_internal_order(owners, neighbours):
    """Internal faces in OpenFOAM upper-triangular order."""
    ordered = array("q")
    run = []
    current_owner = None

    for face_id, neighbour in enumerate(neighbours):
        if neighbour < 0:
            continue
        owner = owners[face_id]
        if current_owner is None:
            current_owner = owner
        if owner != current_owner:
            run.sort(key=lambda idx: neighbours[idx])
            ordered.extend(run)
            run = []
            current_owner = owner
        run.append(face_id)

    if run:
        run.sort(key=lambda idx: neighbours[idx])
        ordered.extend(run)
    return ordered


def write_header(handle, foam_class, object_name):
    handle.write(
        FOAM_HEADER.format(foam_class=foam_class, object_name=object_name)
    )


def write_points(path, points):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        write_header(handle, "vectorField", "points")
        handle.write(f"{len(points)}\n(\n")
        for x, y, z in points:
            handle.write(f"({x:.17g} {y:.17g} {z:.17g})\n")
        handle.write(")\n")


def write_faces(path, face_nodes, ordered_faces):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        write_header(handle, "faceList", "faces")
        handle.write(f"{len(ordered_faces)}\n(\n")
        for face_id in ordered_faces:
            nodes = face_nodes[face_id]
            handle.write(f"{len(nodes)}({' '.join(map(str, nodes))})\n")
        handle.write(")\n")


def write_label_list(path, object_name, values):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        write_header(handle, "labelList", object_name)
        handle.write(f"{len(values)}\n(\n")
        for value in values:
            handle.write(f"{value}\n")
        handle.write(")\n")


def write_boundary(path, patch_names, patch_types, patch_face_lists, n_internal):
    nonempty = [
        patch_id
        for patch_id, faces in enumerate(patch_face_lists)
        if faces
    ]
    start_face = n_internal
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        write_header(handle, "polyBoundaryMesh", "boundary")
        handle.write(f"{len(nonempty)}\n(\n")
        for patch_id in nonempty:
            faces = patch_face_lists[patch_id]
            handle.write(f"    {patch_names[patch_id]}\n")
            handle.write("    {\n")
            handle.write(f"        type            {patch_types[patch_id]};\n")
            handle.write(f"        nFaces          {len(faces)};\n")
            handle.write(f"        startFace       {start_face};\n")
            handle.write("    }\n")
            start_face += len(faces)
        handle.write(")\n")


def write_cell_zones(path, volume_groups, volume_map):
    zones = []
    for zone_name, salome_ids in volume_groups:
        labels = []
        missing = []
        for salome_id in salome_ids:
            cell_id = volume_map.get(salome_id)
            if cell_id is None:
                missing.append(salome_id)
            else:
                labels.append(cell_id)
        if missing:
            raise RuntimeError(
                f"cellZone '{zone_name}' contains {len(missing)} cells that are not "
                "present in the exported mesh."
            )
        zones.append((zone_name, labels))

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        write_header(handle, "regIOobject", "cellZones")
        handle.write(f"{len(zones)}\n(\n")
        for zone_name, labels in zones:
            handle.write(f"    {zone_name}\n")
            handle.write("    {\n")
            handle.write("        type            cellZone;\n")
            handle.write("        cellLabels      List<label>\n")
            handle.write(f"        {len(labels)}\n")
            handle.write("        (\n")
            for label in labels:
                handle.write(f"            {label}\n")
            handle.write("        );\n")
            handle.write("    }\n")
        handle.write(")\n")


def export_mesh(mesh, SMESH, mesh_name, case_dir):
    started = time.time()
    constant_dir = case_dir / "constant"
    constant_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".polyMesh.salome-", dir=constant_dir))

    log(f"Mesh: {mesh_name}")
    log(f"Target case: {case_dir}")
    log(f"Temporary output: {temp_dir}")

    salome_node_ids = list(mesh.GetNodesId())
    if not salome_node_ids:
        salome_node_ids = list(mesh.GetElementsByType(SMESH.NODE))
    points = [tuple(mesh.GetNodeXYZ(node_id)) for node_id in salome_node_ids]
    node_map = {salome_id: point_id for point_id, salome_id in enumerate(salome_node_ids)}
    log(f"Read {len(points)} points")

    (
        patch_names,
        patch_types,
        patch_by_key,
        volume_groups,
    ) = collect_group_data(mesh, SMESH, node_map)
    log(
        f"Found {len(patch_names)} face groups and "
        f"{len(volume_groups)} volume groups"
    )

    salome_volume_ids = list(mesh.GetElementsByType(SMESH.VOLUME))
    volume_map = {
        salome_id: cell_id for cell_id, salome_id in enumerate(salome_volume_ids)
    }

    # Unique face topology and compact per-face data.
    face_index = {}
    face_nodes = []
    owners = array("q")
    neighbours = array("q")
    patch_ids = array("i")

    for cell_id, volume_id in enumerate(salome_volume_ids):
        salome_cell_nodes = mesh.GetElemNodes(volume_id)
        try:
            cell_points = tuple(node_map[node_id] for node_id in salome_cell_nodes)
        except KeyError as exc:
            raise RuntimeError(
                f"Volume {volume_id} references an unknown node: {exc}"
            ) from exc

        found_faces = 0
        for salome_face_nodes in iter_cell_faces(mesh, volume_id):
            found_faces += 1
            try:
                raw_face = tuple(node_map[node_id] for node_id in salome_face_nodes)
            except KeyError as exc:
                raise RuntimeError(
                    f"Volume {volume_id} face references an unknown node: {exc}"
                ) from exc
            key = face_key(raw_face)
            existing = face_index.get(key)

            if existing is None:
                oriented = raw_face
                if VERIFY_FACE_ORIENTATION:
                    oriented = orient_face_outward(oriented, cell_points, points)
                new_face_id = len(face_nodes)
                face_index[key] = new_face_id
                face_nodes.append(oriented)
                owners.append(cell_id)
                neighbours.append(-1)
                patch_ids.append(patch_by_key.get(key, -1))
            else:
                if neighbours[existing] >= 0:
                    raise RuntimeError(
                        "Non-manifold mesh: a face is used by more than two volume "
                        f"cells (SALOME volume {volume_id})."
                    )
                neighbours[existing] = cell_id

        if found_faces == 0:
            raise RuntimeError(
                f"SALOME returned no faces for volume element {volume_id}."
            )

        if PROGRESS_INTERVAL and (cell_id + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - started
            log(
                f"Processed {cell_id + 1}/{len(salome_volume_ids)} cells, "
                f"{len(face_nodes)} unique faces ({elapsed:.0f} s)"
            )

    n_internal = sum(1 for value in neighbours if value >= 0)
    n_boundary = len(face_nodes) - n_internal
    log(
        f"Built topology: {len(salome_volume_ids)} cells, "
        f"{len(face_nodes)} faces ({n_internal} internal, {n_boundary} boundary)"
    )

    # Any grouped face that ended up internal is not a boundary patch here.
    internal_grouped = defaultdict(int)
    for face_id, neighbour in enumerate(neighbours):
        if neighbour >= 0 and patch_ids[face_id] >= 0:
            internal_grouped[patch_names[patch_ids[face_id]]] += 1
    if internal_grouped:
        details = ", ".join(
            f"{name}: {count}" for name, count in sorted(internal_grouped.items())
        )
        log(
            "WARNING: face groups contain internal faces; they are kept internal "
            f"(not converted to baffles): {details}"
        )

    # Append the fallback patch only when it is actually needed.
    default_patch_id = len(patch_names)
    ungrouped_boundary = sum(
        1
        for face_id, neighbour in enumerate(neighbours)
        if neighbour < 0 and patch_ids[face_id] < 0
    )
    if ungrouped_boundary:
        used = set(patch_names)
        default_name = unique_name(
            foam_word(DEFAULT_PATCH_NAME, "defaultFaces"), used
        )
        patch_names.append(default_name)
        patch_types.append(DEFAULT_PATCH_TYPE)
        log(
            f"WARNING: {ungrouped_boundary} boundary faces are not in any SALOME "
            f"face group; writing them to '{default_name}'"
        )

    patch_face_lists = [[] for _ in patch_names]
    for face_id, neighbour in enumerate(neighbours):
        if neighbour >= 0:
            continue
        patch_id = patch_ids[face_id]
        if patch_id < 0:
            patch_id = default_patch_id
        patch_face_lists[patch_id].append(face_id)

    internal_order = make_internal_order(owners, neighbours)
    boundary_order = array("q")
    for faces in patch_face_lists:
        boundary_order.extend(faces)
    ordered_faces = array("q", internal_order)
    ordered_faces.extend(boundary_order)

    if len(ordered_faces) != len(face_nodes):
        raise RuntimeError(
            "Internal exporter error: reordered face count does not match topology."
        )

    ordered_owner = array("q", (owners[face_id] for face_id in ordered_faces))
    ordered_neighbour = array(
        "q", (neighbours[face_id] for face_id in internal_order)
    )

    log("Writing OpenFOAM polyMesh files")
    write_points(temp_dir / "points", points)
    write_faces(temp_dir / "faces", face_nodes, ordered_faces)
    write_label_list(temp_dir / "owner", "owner", ordered_owner)
    write_label_list(temp_dir / "neighbour", "neighbour", ordered_neighbour)
    write_boundary(
        temp_dir / "boundary",
        patch_names,
        patch_types,
        patch_face_lists,
        n_internal,
    )
    if volume_groups:
        write_cell_zones(temp_dir / "cellZones", volume_groups, volume_map)

    # Install atomically enough for a local case, preserving any previous mesh.
    target_dir = constant_dir / "polyMesh"
    backup_dir = None
    if target_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = constant_dir / f"polyMesh.backup-{stamp}"
        suffix = 2
        while backup_dir.exists():
            backup_dir = constant_dir / f"polyMesh.backup-{stamp}-{suffix}"
            suffix += 1
        target_dir.rename(backup_dir)
    temp_dir.rename(target_dir)

    elapsed = time.time() - started
    log(f"DONE: {target_dir}")
    if backup_dir is not None:
        log(f"Previous polyMesh preserved at: {backup_dir}")
    log(f"Export time: {elapsed:.1f} s")
    log("Next commands in the OpenFOAM case:")
    log("  checkMesh")
    log("  checkMesh -allGeometry -allTopology")
    log("  renumberMesh -overwrite   # only after the first checkMesh succeeds")


def main():
    mesh, SMESH, mesh_name = get_selected_mesh()
    case_dir = choose_case_directory()
    if not case_dir.exists():
        raise RuntimeError(f"Case directory does not exist: {case_dir}")
    export_mesh(mesh, SMESH, mesh_name, case_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
