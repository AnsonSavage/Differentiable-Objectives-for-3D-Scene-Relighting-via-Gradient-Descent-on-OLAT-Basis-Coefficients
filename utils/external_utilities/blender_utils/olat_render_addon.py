"""Blender addon for automating One-Light-At-A-Time (OLAT) synthetic dataset rendering."""
import json
import os
import re

import bpy

bl_info = {
    "name": "OLAT Render Tools",
    "author": "GitHub Copilot",
    "version": (1, 2),
    "blender": (3, 0, 0),
    "location": "Properties > Render > OLAT Render",
    "description": "Tools for rendering One-Light-At-A-Time (OLAT) datasets",
    "category": "Render",
}


def get_emissive_material_indices(obj: bpy.types.Object) -> list[int]:
    """Return a list of material slot indices containing emissive shaders on the object.

    Args:
        obj: Blender object to inspect.

    Returns:
        List of integer material slot indices.
    """
    if obj.type != "MESH":
        return []

    indices = []
    for i, slot in enumerate(obj.material_slots):
        mat = slot.material
        if mat and mat.use_nodes:
            is_emissive = False
            for node in mat.node_tree.nodes:
                strength_input = None
                color_input = None

                if node.type == "BSDF_PRINCIPLED":
                    strength_input = node.inputs.get("Emission Strength") or node.inputs.get("Emission")
                    color_input = node.inputs.get("Emission") or node.inputs.get("Emission Color")
                elif node.type == "EMISSION":
                    strength_input = node.inputs.get("Strength")
                    color_input = node.inputs.get("Color")

                if strength_input and color_input:
                    has_strength = strength_input.is_linked or strength_input.default_value > 0
                    is_not_black = color_input.is_linked or any(color_input.default_value[:3])

                    if has_strength and is_not_black:
                        is_emissive = True
                        break
            if is_emissive:
                indices.append(i)
    return indices


def is_world_light(world: bpy.types.World | None) -> bool:
    """Check if the world environment is emitting light.

    Args:
        world: Blender World data block.

    Returns:
        True if world has non-black/non-zero background emission.
    """
    if not world or not world.use_nodes:
        return False

    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            strength = node.inputs.get("Strength")
            color = node.inputs.get("Color")
            if strength and color:
                has_strength = strength.is_linked or strength.default_value > 0
                is_not_black = color.is_linked or any(color.default_value[:3])
                if has_strength and is_not_black:
                    return True
    return False


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be filesystem-safe for render filenames.

    Args:
        name: Raw identifier string.

    Returns:
        Sanitized string with illegal characters replaced by underscores.
    """
    return re.sub(r'[<>:"/\\|?*\s]', "_", name)


class OLAT_OT_DetectLights(bpy.types.Operator):
    """Detect lights and mesh lights in the scene and mark them for OLAT rendering."""

    bl_idname = "olat.detect_lights"
    bl_label = "Detect Lights"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        """Execute light detection across world and scene objects."""
        count = 0

        # World light detection
        if context.scene.world and is_world_light(context.scene.world):
            w = context.scene.world
            w.olat_is_light = True
            if not w.olat_enabled:
                w.olat_enabled = True
            if not w.olat_optimizable:
                w.olat_optimizable = True
            count += 1

        # Iterate over a snapshot of scene objects
        scene_objects = list(context.scene.objects)

        for obj in scene_objects:
            if obj.type == "LIGHT":
                obj.olat_is_light = True
                if not obj.olat_enabled:
                    obj.olat_enabled = True
                if not obj.olat_optimizable:
                    obj.olat_optimizable = True
                count += 1
                continue

            if obj.type == "MESH":
                emissive_indices = get_emissive_material_indices(obj)

                if not emissive_indices:
                    continue

                # Separate mesh if it has both emissive and non-emissive materials
                is_mixed = len(emissive_indices) < len(obj.material_slots)

                if is_mixed:
                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    context.view_layer.objects.active = obj

                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="DESELECT")

                    for idx in emissive_indices:
                        obj.active_material_index = idx
                        bpy.ops.object.material_slot_select()

                    try:
                        bpy.ops.mesh.separate(type="SELECTED")
                    except Exception as e:
                        print(f"Separation failed for {obj.name}: {e}")
                        bpy.ops.object.mode_set(mode="OBJECT")
                        continue

                    bpy.ops.object.mode_set(mode="OBJECT")

                    selected = context.selected_objects
                    new_parts = [o for o in selected if o != obj]

                    for part in new_parts:
                        part.name = f"{obj.name}_Emissive"
                        part.olat_is_light = True
                        part.olat_enabled = True
                        part.olat_optimizable = True
                        count += 1

                        # Remove non-emissive material slots in reverse order
                        for i in range(len(part.material_slots) - 1, -1, -1):
                            if i not in emissive_indices:
                                part.data.materials.pop(index=i)

                    # Remove emissive material slots from the original non-emissive object
                    context.view_layer.objects.active = obj
                    for i in range(len(obj.material_slots) - 1, -1, -1):
                        if i in emissive_indices:
                            obj.data.materials.pop(index=i)

                else:
                    obj.olat_is_light = True
                    if not obj.olat_enabled:
                        obj.olat_enabled = True
                    if not obj.olat_optimizable:
                        obj.olat_optimizable = True
                    count += 1

        self.report({"INFO"}, f"Detected {count} lights.")
        return {"FINISHED"}


class OLAT_OT_ClearLights(bpy.types.Operator):
    """Unmark all objects and world as lights."""

    bl_idname = "olat.clear_lights"
    bl_label = "Clear All"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        """Execute light marking clearance."""
        if context.scene.world:
            context.scene.world.olat_optimizable = False
            context.scene.world.olat_is_light = False
            context.scene.world.olat_enabled = False

        for obj in context.scene.objects:
            obj.olat_optimizable = False
            obj.olat_is_light = False
            obj.olat_enabled = False
        return {"FINISHED"}


class OLAT_OT_CreateDomeLights(bpy.types.Operator):
    """Create a dome array of quad mesh lights."""

    bl_idname = "olat.create_dome_lights"
    bl_label = "Create Dome Lights"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        """Build and populate hemisphere dome of emission planes."""
        scene = context.scene
        subdiv_level = scene.olat_dome_subdiv_level

        collection_name = "Dome_Lights"
        collection = bpy.data.collections.new(collection_name)
        scene.collection.children.link(collection)

        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align="WORLD", location=(0, 0, 0))
        cube = context.active_object

        for col in cube.users_collection:
            col.objects.unlink(cube)
        collection.objects.link(cube)

        bpy.ops.object.mode_set(mode="EDIT")

        for _ in range(subdiv_level):
            bpy.ops.mesh.subdivide()

        bpy.ops.transform.tosphere(value=1)

        bpy.ops.mesh.select_all(action="DESELECT")

        import bmesh
        bm = bmesh.from_edit_mesh(cube.data)
        bm.verts.ensure_lookup_table()

        # Select and delete lower hemisphere vertices (Z < 0)
        verts_to_delete = [v for v in bm.verts if v.co.z < -0.001]
        for v in verts_to_delete:
            v.select = True

        bm.select_flush(True)
        bmesh.update_edit_mesh(cube.data)

        bpy.ops.mesh.delete(type="VERT")

        # Separate faces into individual light objects
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.edge_split(type="EDGE")
        bpy.ops.mesh.separate(type="LOOSE")

        bpy.ops.object.mode_set(mode="OBJECT")

        selected_objects = context.selected_objects

        # Create emission shader
        mat = bpy.data.materials.new(name="DomeLight_Emission")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        nodes.clear()

        node_emission = nodes.new(type="ShaderNodeEmission")
        node_emission.inputs["Strength"].default_value = 1.0
        node_emission.inputs["Color"].default_value = (1, 1, 1, 1)

        node_output = nodes.new(type="ShaderNodeOutputMaterial")
        links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])

        for i, obj in enumerate(selected_objects):
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            obj.name = f"DomeLight_{i:03d}"
            obj.olat_is_light = True
            obj.olat_enabled = True
            obj.olat_optimizable = True

        self.report({"INFO"}, f"Created dome with {len(selected_objects)} lights.")
        return {"FINISHED"}


class OLAT_OT_Render(bpy.types.Operator):
    """Render selected individual lights to OpenEXR passes."""

    bl_idname = "olat.render"
    bl_label = "Render OLAT"

    def execute(self, context):
        """Execute OLAT rendering passes and output metadata JSON."""
        scene = context.scene
        output_dir = bpy.path.abspath(scene.olat_output_dir)

        if not output_dir:
            self.report({"ERROR"}, "Output directory not set")
            return {"CANCELLED"}

        os.makedirs(output_dir, exist_ok=True)
        optimizable_output_dir = os.path.join(output_dir, "optimizable_lights")
        os.makedirs(optimizable_output_dir, exist_ok=True)

        all_lights = [obj for obj in scene.objects if obj.olat_is_light and obj.olat_enabled]

        original_world = scene.world
        if original_world and original_world.olat_is_light and original_world.olat_enabled:
            all_lights.append(original_world)

        if not all_lights:
            self.report({"WARNING"}, "No enabled lights detected. Run 'Detect Lights' or enable some lights.")
            return {"CANCELLED"}

        optimizable_lights = [l for l in all_lights if l.olat_optimizable]
        static_lights = [l for l in all_lights if not l.olat_optimizable]

        original_visibility = {obj.name: obj.hide_render for obj in scene.objects}
        original_filepath = scene.render.filepath
        original_format = scene.render.image_settings.file_format

        scene.render.image_settings.file_format = "OPEN_EXR"

        light_mapping = {}

        try:
            # Hide all lights initially
            for obj in scene.objects:
                if obj.olat_is_light:
                    obj.hide_render = True

            scene.world = None

            # 1. Render Optimizable Lights (OLAT passes)
            total = len(optimizable_lights)
            optimizable_object_names = {l.name for l in optimizable_lights if l != original_world}

            for i, active_light in enumerate(optimizable_lights):
                print(f"Rendering pass {i + 1}/{total}: {active_light.name}")

                if active_light == original_world:
                    scene.world = original_world
                    safe_name = sanitize_filename(active_light.name)
                    if safe_name in optimizable_object_names:
                        safe_name = f"{safe_name}_Environment"
                        if safe_name in optimizable_object_names:
                            safe_name = f"{safe_name}_Global"
                else:
                    active_light.hide_render = False
                    safe_name = sanitize_filename(active_light.name)

                filename = f"olat_{safe_name}.exr"
                scene.render.filepath = os.path.join(optimizable_output_dir, filename)
                light_mapping[filename] = active_light.name

                bpy.ops.render.render(write_still=True)

                if active_light == original_world:
                    scene.world = None
                else:
                    active_light.hide_render = True

            # Save mapping metadata
            mapping_path = os.path.join(output_dir, "olat_metadata.json")
            with open(mapping_path, "w") as f:
                json.dump({
                    "light_mapping": light_mapping,
                    "static_lights_count": len(static_lights),
                }, f, indent=4)

            # 2. Render Static Lights (combined background pass)
            if static_lights:
                print(f"Rendering static lights pass with {len(static_lights)} lights")

                for light in static_lights:
                    if light == original_world:
                        scene.world = original_world
                    else:
                        light.hide_render = False

                filename = "base_lighting.exr"
                scene.render.filepath = os.path.join(output_dir, filename)
                bpy.ops.render.render(write_still=True)

                for light in static_lights:
                    if light == original_world:
                        scene.world = None
                    else:
                        light.hide_render = True

        finally:
            # Restore initial scene state
            for name, hidden in original_visibility.items():
                if name in scene.objects:
                    scene.objects[name].hide_render = hidden

            scene.world = original_world
            scene.render.filepath = original_filepath
            scene.render.image_settings.file_format = original_format

        self.report({"INFO"}, f"Rendered {len(optimizable_lights)} OLAT passes and {'1 static pass' if static_lights else '0 static passes'}.")
        return {"FINISHED"}


def update_collection_enabled(self, context) -> None:
    """Propagate enabled state to child objects and sub-collections."""
    val = self.olat_enabled
    for obj in self.objects:
        if obj.olat_is_light:
            obj.olat_enabled = val
    for child in self.children:
        child.olat_enabled = val


def update_collection_optimizable(self, context) -> None:
    """Propagate optimizable flag to child objects and sub-collections."""
    val = self.olat_optimizable
    for obj in self.objects:
        if obj.olat_is_light:
            obj.olat_optimizable = val
    for child in self.children:
        child.olat_optimizable = val


class OLAT_PT_Panel(bpy.types.Panel):
    """UI panel for OLAT render addon in the Render properties tab."""

    bl_label = "OLAT Render"
    bl_idname = "OLAT_PT_main_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw_collection_recursive(self, layout, collection, level, parent_optimizable=True):
        """Recursively draw light hierarchy and toggles in the panel layout."""
        has_lights = False
        for obj in collection.all_objects:
            if obj.olat_is_light:
                has_lights = True
                break

        if not has_lights:
            return

        row = layout.row()
        split = row.split(factor=0.6)

        # Hierarchy tree (left)
        left_row = split.row()
        if level > 0:
            for _ in range(level):
                left_row.label(text="", icon="BLANK1")

        icon = "TRIA_DOWN" if collection.olat_expanded else "TRIA_RIGHT"
        left_row.prop(collection, "olat_expanded", icon=icon, icon_only=True, emboss=False)
        left_row.label(text=collection.name, icon="OUTLINER_COLLECTION")

        # Controls (right)
        right_row = split.row()
        right_row.alignment = "RIGHT"
        right_row.prop(collection, "olat_enabled", text="Enabled")

        opt_row = right_row.row()
        opt_row.prop(collection, "olat_optimizable", text="Optimizable")
        opt_row.enabled = parent_optimizable and collection.olat_enabled

        current_optimizable = parent_optimizable and collection.olat_optimizable

        if collection.olat_expanded:
            for obj in collection.objects:
                if obj.olat_is_light:
                    row = layout.row()
                    split = row.split(factor=0.6)

                    left_row = split.row()
                    for _ in range(level + 1):
                        left_row.label(text="", icon="BLANK1")

                    left_row.label(text=" ", icon="BLANK1")
                    left_row.label(text=obj.name, icon="LIGHT" if obj.type == "LIGHT" else "MESH_CUBE")

                    right_row = split.row()
                    right_row.alignment = "RIGHT"
                    right_row.prop(obj, "olat_enabled", text="Enabled")

                    opt_row = right_row.row()
                    opt_row.prop(obj, "olat_optimizable", text="Optimizable")
                    opt_row.enabled = current_optimizable and obj.olat_enabled

            for child in collection.children:
                self.draw_collection_recursive(layout, child, level + 1, parent_optimizable=current_optimizable)

    def draw(self, context):
        """Draw panel widgets and buttons."""
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "olat_output_dir", text="Output Directory")

        row = layout.row()
        row.operator("olat.detect_lights", text="Refresh / Detect Lights")
        row.operator("olat.clear_lights")

        layout.label(text="Lights Hierarchy:")

        box = layout.box()
        col = box.column()

        # World light entry
        if scene.world and scene.world.olat_is_light:
            row = col.row()
            split = row.split(factor=0.6)

            left_row = split.row()
            left_row.label(text="World Environment", icon="WORLD")

            right_row = split.row()
            right_row.alignment = "RIGHT"
            right_row.prop(scene.world, "olat_enabled", text="Enabled")

            opt_row = right_row.row()
            opt_row.prop(scene.world, "olat_optimizable", text="Optimizable")
            opt_row.enabled = scene.world.olat_enabled

            col.separator()

        self.draw_collection_recursive(col, scene.collection, 0)

        layout.separator()
        layout.operator("olat.render", icon="RENDER_STILL")

        layout.separator()
        layout.label(text="Utilities:")
        box = layout.box()
        box.prop(scene, "olat_dome_subdiv_level")
        box.operator("olat.create_dome_lights")


def register():
    """Register Blender addon classes and properties."""
    bpy.utils.register_class(OLAT_OT_DetectLights)
    bpy.utils.register_class(OLAT_OT_ClearLights)
    bpy.utils.register_class(OLAT_OT_CreateDomeLights)
    bpy.utils.register_class(OLAT_OT_Render)
    bpy.utils.register_class(OLAT_PT_Panel)

    bpy.types.Scene.olat_output_dir = bpy.props.StringProperty(
        name="Output Directory",
        description="Directory to save rendered EXR files",
        default="//examples/EXAMPLE_OLATS",
        subtype="DIR_PATH",
    )

    bpy.types.Scene.olat_dome_subdiv_level = bpy.props.IntProperty(
        name="Subdivision Level",
        description="Subdivision level for dome lights",
        default=2,
        min=1,
        max=4,
    )

    bpy.types.Object.olat_optimizable = bpy.props.BoolProperty(
        name="OLAT Optimizable",
        description="Include this object in OLAT render",
        default=True,
    )

    bpy.types.Object.olat_enabled = bpy.props.BoolProperty(
        name="OLAT Enabled",
        description="Include this object in any render pass",
        default=True,
    )

    bpy.types.Object.olat_is_light = bpy.props.BoolProperty(
        name="OLAT Is Light",
        description="Identified as a light source",
        default=False,
    )

    bpy.types.World.olat_optimizable = bpy.props.BoolProperty(
        name="OLAT Optimizable",
        description="Include world in OLAT render",
        default=True,
    )

    bpy.types.World.olat_enabled = bpy.props.BoolProperty(
        name="OLAT Enabled",
        description="Include world in any render pass",
        default=True,
    )

    bpy.types.World.olat_is_light = bpy.props.BoolProperty(
        name="OLAT Is Light",
        description="Identified as a light source",
        default=False,
    )

    bpy.types.Collection.olat_expanded = bpy.props.BoolProperty(
        name="Expanded",
        default=True,
    )
    bpy.types.Collection.olat_enabled = bpy.props.BoolProperty(
        name="Enabled",
        default=True,
        update=update_collection_enabled,
    )
    bpy.types.Collection.olat_optimizable = bpy.props.BoolProperty(
        name="Optimizable",
        default=True,
        update=update_collection_optimizable,
    )


def unregister():
    """Unregister Blender addon classes and properties."""
    bpy.utils.unregister_class(OLAT_OT_DetectLights)
    bpy.utils.unregister_class(OLAT_OT_ClearLights)
    bpy.utils.unregister_class(OLAT_OT_CreateDomeLights)
    bpy.utils.unregister_class(OLAT_OT_Render)
    bpy.utils.unregister_class(OLAT_PT_Panel)

    del bpy.types.Scene.olat_output_dir
    del bpy.types.Scene.olat_dome_subdiv_level

    del bpy.types.Object.olat_optimizable
    del bpy.types.Object.olat_enabled
    del bpy.types.Object.olat_is_light

    del bpy.types.World.olat_optimizable
    del bpy.types.World.olat_enabled
    del bpy.types.World.olat_is_light

    del bpy.types.Collection.olat_expanded
    del bpy.types.Collection.olat_enabled
    del bpy.types.Collection.olat_optimizable


if __name__ == "__main__":
    register()