import bpy
import os
import json
import re

bl_info = {
    "name": "OLAT Render Tools",
    "author": "GitHub Copilot",
    "version": (1, 2),
    "blender": (3, 0, 0),
    "location": "Properties > Render > OLAT Render",
    "description": "Tools for rendering One-Light-At-A-Time (OLAT) datasets",
    "category": "Render",
}

def get_emissive_material_indices(obj):
    """Return a list of indices of emissive materials on the object."""
    if obj.type != 'MESH':
        return []
    
    indices = []
    for i, slot in enumerate(obj.material_slots):
        mat = slot.material
        if mat and mat.use_nodes:
            is_emissive = False
            for node in mat.node_tree.nodes:
                strength_input = None
                color_input = None

                # --- Case 1: Principled BSDF Shader ---
                if node.type == 'BSDF_PRINCIPLED':
                    strength_input = node.inputs.get('Emission Strength') or node.inputs.get('Emission')
                    color_input = node.inputs.get('Emission') or node.inputs.get('Emission Color')

                # --- Case 2: Emission Shader ---
                elif node.type == 'EMISSION':
                    strength_input = node.inputs.get('Strength')
                    color_input = node.inputs.get('Color')

                # --- Process the found node ---
                if strength_input and color_input:
                    has_strength = strength_input.is_linked or strength_input.default_value > 0
                    is_not_black = color_input.is_linked or any(color_input.default_value[:3])

                    if has_strength and is_not_black:
                        is_emissive = True
                        break
            if is_emissive:
                indices.append(i)
    return indices

def is_world_light(world):
    """Check if the world is emitting light."""
    if not world or not world.use_nodes:
        return False
    
    for node in world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            strength = node.inputs.get('Strength')
            color = node.inputs.get('Color')
            if strength and color:
                has_strength = strength.is_linked or strength.default_value > 0
                is_not_black = color.is_linked or any(color.default_value[:3])
                if has_strength and is_not_black:
                    return True
    return False

def sanitize_filename(name):
    """Sanitize a string to be safe for filenames."""
    # Replace invalid characters with underscore
    # Windows invalid: < > : " / \ | ? *
    # Also replace spaces for cleaner filenames
    s = re.sub(r'[<>:"/\\|?*\s]', '_', name)
    return s

class OLAT_OT_DetectLights(bpy.types.Operator):
    """Detects lights and mesh lights in the scene and marks them"""
    bl_idname = "olat.detect_lights"
    bl_label = "Detect Lights"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        count = 0
        
        # Check World
        if context.scene.world and is_world_light(context.scene.world):
            w = context.scene.world
            w.olat_is_light = True
            # Default to enabled and optimizable if it wasn't tracked before
            if not w.olat_enabled:
                w.olat_enabled = True
            if not w.olat_optimizable:
                w.olat_optimizable = True
            count += 1
            
        # Snapshot of objects to iterate safely while modifying scene
        scene_objects = list(context.scene.objects)
        
        for obj in scene_objects:
            if obj.type == 'LIGHT':
                obj.olat_is_light = True
                if not obj.olat_enabled: obj.olat_enabled = True
                if not obj.olat_optimizable: obj.olat_optimizable = True
                count += 1
                continue
                
            if obj.type == 'MESH':
                emissive_indices = get_emissive_material_indices(obj)
                
                if not emissive_indices:
                    continue
                    
                # Check if we need to separate
                # We separate if there are non-emissive materials AND emissive materials
                # Or if the object has multiple materials and only some are emissive.
                
                is_mixed = len(emissive_indices) < len(obj.material_slots)
                
                if is_mixed:
                    # Perform separation
                    # Deselect all
                    bpy.ops.object.select_all(action='DESELECT')
                    
                    # Select object and make active
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    
                    # Go to Edit Mode
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='DESELECT')
                    
                    # Select faces for each emissive index
                    for idx in emissive_indices:
                        obj.active_material_index = idx
                        bpy.ops.object.material_slot_select()
                        
                    try:
                        bpy.ops.mesh.separate(type='SELECTED')
                    except Exception as e:
                        # Might fail if nothing selected
                        print(f"Separation failed for {obj.name}: {e}")
                        bpy.ops.object.mode_set(mode='OBJECT')
                        continue
                        
                    bpy.ops.object.mode_set(mode='OBJECT')
                    
                    # Now we have multiple selected objects.
                    # The original object (active) contains the unselected parts (non-emissive).
                    # The new objects contain the selected parts (emissive).
                    
                    selected = context.selected_objects
                    new_parts = [o for o in selected if o != obj]
                    
                    for part in new_parts:
                        part.name = f"{obj.name}_Emissive"
                        part.olat_is_light = True
                        part.olat_enabled = True
                        part.olat_optimizable = True
                        count += 1
                        
                        # Remove non-emissive materials from the new part
                        # We iterate backwards to keep indices valid for the slots we haven't checked yet
                        
                        # Remove non-emissive materials from the new part
                        # We iterate backwards to keep indices valid for the slots we haven't checked yet
                        for i in range(len(part.material_slots) - 1, -1, -1):
                            if i not in emissive_indices:
                                part.data.materials.pop(index=i)
                    
                    # Remove the emissive material(s) from the original mesh
                    context.view_layer.objects.active = obj
                    for i in range(len(obj.material_slots) - 1, -1, -1):
                        if i in emissive_indices:
                            obj.data.materials.pop(index=i)
                        
                else:
                    # Fully emissive (or at least no non-emissive materials found)
                    obj.olat_is_light = True
                    if not obj.olat_enabled: obj.olat_enabled = True
                    if not obj.olat_optimizable: obj.olat_optimizable = True
                    count += 1
        
        self.report({'INFO'}, f"Detected {count} lights.")
        return {'FINISHED'}

class OLAT_OT_ClearLights(bpy.types.Operator):
    """Unmarks all objects as lights/optimizable"""
    bl_idname = "olat.clear_lights"
    bl_label = "Clear All"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.scene.world:
            context.scene.world.olat_optimizable = False
            context.scene.world.olat_is_light = False
            context.scene.world.olat_enabled = False
            
        for obj in context.scene.objects:
            obj.olat_optimizable = False
            obj.olat_is_light = False
            obj.olat_enabled = False
        return {'FINISHED'}

class OLAT_OT_CreateDomeLights(bpy.types.Operator):
    """Create a dome of quad lights"""
    bl_idname = "olat.create_dome_lights"
    bl_label = "Create Dome Lights"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        subdiv_level = scene.olat_dome_subdiv_level
        
        # Create Collection
        collection_name = "Dome_Lights"
        collection = bpy.data.collections.new(collection_name)
        scene.collection.children.link(collection)
        
        # Create Cube
        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        cube = context.active_object
        
        # Move cube to collection
        # Unlink from current collections
        for col in cube.users_collection:
            col.objects.unlink(cube)
        collection.objects.link(cube)
        
        # Edit Mode operations
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Subdivide
        for i in range(subdiv_level):
            bpy.ops.mesh.subdivide()
            
        # To Sphere
        bpy.ops.transform.tosphere(value=1)
        
        # Delete bottom half
        # We need to select vertices with Z < -epsilon
        # Deselect all
        bpy.ops.mesh.select_all(action='DESELECT')
        
        # We need to switch to object mode to access data, or use bmesh
        # Using bmesh is safer for selection logic
        import bmesh
        bm = bmesh.from_edit_mesh(cube.data)
        bm.verts.ensure_lookup_table()
        
        # Select vertices with z < -0.001
        verts_to_delete = [v for v in bm.verts if v.co.z < -0.001]
        for v in verts_to_delete:
            v.select = True
            
        # Flush selection
        bm.select_flush(True) 
        bmesh.update_edit_mesh(cube.data)
        
        # Delete vertices
        bpy.ops.mesh.delete(type='VERT')
        
        # Now separate faces
        # Select All
        bpy.ops.mesh.select_all(action='SELECT')
        # Edge Split to disconnect faces
        bpy.ops.mesh.edge_split(type='EDGE')
        # Separate by loose parts
        bpy.ops.mesh.separate(type='LOOSE')
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Now we have multiple objects selected.
        # The original 'cube' is one of them.
        # We need to apply material to all selected objects.
        
        selected_objects = context.selected_objects
        
        # Create Material
        mat = bpy.data.materials.new(name="DomeLight_Emission")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create Emission node
        node_emission = nodes.new(type='ShaderNodeEmission')
        node_emission.inputs['Strength'].default_value = 1.0
        node_emission.inputs['Color'].default_value = (1, 1, 1, 1)
        
        # Create Output node
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # Link
        links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])
        
        # Assign material to all objects
        for i, obj in enumerate(selected_objects):
            # Clear existing materials
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            
            # Rename for clarity
            obj.name = f"DomeLight_{i:03d}"
            
            # Ensure they are tracked by OLAT
            obj.olat_is_light = True
            obj.olat_enabled = True
            obj.olat_optimizable = True

        self.report({'INFO'}, f"Created dome with {len(selected_objects)} lights.")
        return {'FINISHED'}

class OLAT_OT_Render(bpy.types.Operator):
    """Render selected lights to EXR (OLAT + Static pass)"""
    bl_idname = "olat.render"
    bl_label = "Render OLAT"
    
    def execute(self, context):
        scene = context.scene
        output_dir = bpy.path.abspath(scene.olat_output_dir)
        
        if not output_dir:
            self.report({'ERROR'}, "Output directory not set")
            return {'CANCELLED'}
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect lights based on the detected flag AND enabled flag
        all_lights = [obj for obj in scene.objects if obj.olat_is_light and obj.olat_enabled]
        
        # Add World if applicable
        original_world = scene.world
        if original_world and original_world.olat_is_light and original_world.olat_enabled:
            all_lights.append(original_world)
        
        if not all_lights:
            self.report({'WARNING'}, "No enabled lights detected. Run 'Detect Lights' or enable some lights.")
            return {'CANCELLED'}

        # Split into optimizable and static
        optimizable_lights = [l for l in all_lights if l.olat_optimizable]
        static_lights = [l for l in all_lights if not l.olat_optimizable]

        # Save original state
        original_visibility = {obj.name: obj.hide_render for obj in scene.objects}
        original_filepath = scene.render.filepath
        original_format = scene.render.image_settings.file_format
        
        # Configure Render Settings
        scene.render.image_settings.file_format = 'OPEN_EXR'
        
        # Mapping dictionary: sanitized_filename -> original_blender_name
        light_mapping = {}
        
        try:
            # Hide ALL lights first (even disabled ones, to be safe)
            # We iterate over all objects that are marked as lights
            for obj in scene.objects:
                if obj.olat_is_light:
                    obj.hide_render = True
            
            # Hide World initially
            scene.world = None
                
            # 1. Render Optimizable Lights (OLAT)
            total = len(optimizable_lights)
            
            # Get names of all optimizable OBJECTS (exclude world) to check for collisions
            optimizable_object_names = {l.name for l in optimizable_lights if l != original_world}
            
            for i, active_light in enumerate(optimizable_lights):
                print(f"Rendering pass {i + 1}/{total}: {active_light.name}")
                
                # Enable current light
                if active_light == original_world:
                    scene.world = original_world
                    
                    # Handle name collision for World
                    # If an object has the same name as the world, rename the world output
                    safe_name = sanitize_filename(active_light.name)
                    if safe_name in optimizable_object_names:
                        safe_name = f"{safe_name}_Environment"
                        # Double check in case an object is named "World_Environment"
                        if safe_name in optimizable_object_names:
                            safe_name = f"{safe_name}_Global"
                else:
                    active_light.hide_render = False
                    safe_name = sanitize_filename(active_light.name)
                
                # Set output path
                # Consistent prefix "olat_" and suffix is the light name
                filename = f"olat_{safe_name}.exr"
                scene.render.filepath = os.path.join(output_dir, filename)
                
                # Store mapping
                light_mapping[filename] = active_light.name
                
                # Render
                bpy.ops.render.render(write_still=True)
                
                # Disable again for next pass
                if active_light == original_world:
                    scene.world = None
                else:
                    active_light.hide_render = True

            # Save mapping to JSON
            mapping_path = os.path.join(output_dir, "olat_metadata.json")
            with open(mapping_path, 'w') as f:
                json.dump({
                    "light_mapping": light_mapping,
                    "static_lights_count": len(static_lights)
                }, f, indent=4)

            # 2. Render Static Lights (Combined)
            if static_lights:
                print(f"Rendering static lights pass with {len(static_lights)} lights")
                
                # Enable all static lights
                for light in static_lights:
                    if light == original_world:
                        scene.world = original_world
                    else:
                        light.hide_render = False
                
                # Set output path
                filename = "non_optimized_lights.exr"
                scene.render.filepath = os.path.join(output_dir, filename)
                
                # Render
                bpy.ops.render.render(write_still=True)
                
                # Disable again (cleanup)
                for light in static_lights:
                    if light == original_world:
                        scene.world = None
                    else:
                        light.hide_render = True
                
        finally:
            # Restore state
            for name, hidden in original_visibility.items():
                if name in scene.objects:
                    scene.objects[name].hide_render = hidden
            
            scene.world = original_world
            scene.render.filepath = original_filepath
            scene.render.image_settings.file_format = original_format
            
        self.report({'INFO'}, f"Rendered {len(optimizable_lights)} OLAT passes and {'1 static pass' if static_lights else '0 static passes'}.")
        return {'FINISHED'}

def update_collection_enabled(self, context):
    val = self.olat_enabled
    # Propagate to objects in this collection
    for obj in self.objects:
        if obj.olat_is_light:
            obj.olat_enabled = val
    # Propagate to child collections
    for child in self.children:
        child.olat_enabled = val

def update_collection_optimizable(self, context):
    val = self.olat_optimizable
    # Propagate to objects in this collection
    for obj in self.objects:
        if obj.olat_is_light:
            obj.olat_optimizable = val
    # Propagate to child collections
    for child in self.children:
        child.olat_optimizable = val

class OLAT_PT_Panel(bpy.types.Panel):
    bl_label = "OLAT Render"
    bl_idname = "OLAT_PT_main_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw_collection_recursive(self, layout, collection, level, parent_optimizable=True):
        # Check if this collection or its children contain any lights
        has_lights = False
        for obj in collection.all_objects:
            if obj.olat_is_light:
                has_lights = True
                break
        
        if not has_lights:
            return

        row = layout.row()
        
        # Split row into Tree (Left) and Controls (Right)
        # Adjust factor to give enough space for deep trees
        split = row.split(factor=0.6)
        
        # --- Left Side: Tree Structure ---
        left_row = split.row()
        
        # Indentation
        if level > 0:
            # Use a split for indentation to ensure consistent alignment of the name
            # But recursive splits get messy. 
            # Simple spacer icons or labels work best for tree views.
            for _ in range(level):
                left_row.label(text="", icon='BLANK1') 
        
        # Expand/Collapse icon
        icon = 'TRIA_DOWN' if collection.olat_expanded else 'TRIA_RIGHT'
        left_row.prop(collection, "olat_expanded", icon=icon, icon_only=True, emboss=False)
        
        # Name
        left_row.label(text=collection.name, icon="OUTLINER_COLLECTION")
        
        # --- Right Side: Controls ---
        right_row = split.row()
        right_row.alignment = 'RIGHT'
        
        # Enabled Checkbox
        right_row.prop(collection, "olat_enabled", text="Enabled")
        
        # Optimizable Checkbox
        # Only enabled if parent is optimizable AND this collection is enabled
        # (Though usually "Enabled" toggle controls visibility, "Optimizable" controls the property)
        # User requirement: "if a collection is marked as non optimizable, then a subelement... shouldn't be able to be marked"
        
        opt_row = right_row.row()
        opt_row.prop(collection, "olat_optimizable", text="Optimizable")
        # Disable if parent forbids it, or if self is disabled (optional, but good UX)
        opt_row.enabled = parent_optimizable and collection.olat_enabled
        
        # Determine optimizable state for children
        # If this collection is NOT optimizable, children cannot be either.
        # If this collection IS optimizable, children can be (unless they disable it themselves).
        # Note: The checkbox value `collection.olat_optimizable` stores the user intent.
        # But the effective state passed down depends on parent too.
        # Actually, if parent is False, this collection's checkbox is disabled.
        # If it was previously True, it might still be True in data, but effectively False.
        # So we should pass down (parent_optimizable and collection.olat_optimizable).
        
        current_optimizable = parent_optimizable and collection.olat_optimizable

        if collection.olat_expanded:
            # Draw objects in this collection
            for obj in collection.objects:
                if obj.olat_is_light:
                    row = layout.row()
                    split = row.split(factor=0.6)
                    
                    # Left: Tree
                    left_row = split.row()
                    for _ in range(level + 1):
                        left_row.label(text="", icon='BLANK1')
                    
                    left_row.label(text=" ", icon='BLANK1') # Indent for the expand icon space
                    left_row.label(text=obj.name, icon="LIGHT" if obj.type=='LIGHT' else "MESH_CUBE")
                    
                    # Right: Controls
                    right_row = split.row()
                    right_row.alignment = 'RIGHT'
                    
                    # Enabled
                    right_row.prop(obj, "olat_enabled", text="Enabled")
                    
                    # Optimizable
                    opt_row = right_row.row()
                    opt_row.prop(obj, "olat_optimizable", text="Optimizable")
                    # Disabled if parent collection is not optimizable OR object is disabled
                    opt_row.enabled = current_optimizable and obj.olat_enabled
            
            # Draw children collections
            for child in collection.children:
                self.draw_collection_recursive(layout, child, level + 1, parent_optimizable=current_optimizable)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.prop(scene, "olat_output_dir", text="Output Directory")
        
        row = layout.row()
        row.operator("olat.detect_lights", text="Refresh / Detect Lights")
        row.operator("olat.clear_lights")
        
        layout.label(text="Lights Hierarchy:")
        
        box = layout.box()
        col = box.column()
        
        # --- World Light ---
        if scene.world and scene.world.olat_is_light:
            row = col.row()
            split = row.split(factor=0.6)
            
            # Left: Name
            left_row = split.row()
            left_row.label(text="World Environment", icon="WORLD")
            
            # Right: Controls
            right_row = split.row()
            right_row.alignment = 'RIGHT'
            
            # Enabled
            right_row.prop(scene.world, "olat_enabled", text="Enabled")
            
            # Optimizable
            opt_row = right_row.row()
            opt_row.prop(scene.world, "olat_optimizable", text="Optimizable")
            opt_row.enabled = scene.world.olat_enabled
            
            col.separator()
        
        # Start from master collection
        self.draw_collection_recursive(col, scene.collection, 0)
            
        layout.separator()
        layout.operator("olat.render", icon="RENDER_STILL")
        
        layout.separator()
        layout.label(text="Utilities:")
        box = layout.box()
        box.prop(scene, "olat_dome_subdiv_level")
        box.operator("olat.create_dome_lights")

def register():
    bpy.utils.register_class(OLAT_OT_DetectLights)
    bpy.utils.register_class(OLAT_OT_ClearLights)
    bpy.utils.register_class(OLAT_OT_CreateDomeLights)
    bpy.utils.register_class(OLAT_OT_Render)
    bpy.utils.register_class(OLAT_PT_Panel)
    
    bpy.types.Scene.olat_output_dir = bpy.props.StringProperty(
        name="Output Directory",
        description="Directory to save rendered EXR files",
        default="//rendered_lights",
        subtype='DIR_PATH',
        
    )
    
    bpy.types.Scene.olat_dome_subdiv_level = bpy.props.IntProperty(
        name="Subdivision Level",
        description="Subdivision level for dome lights",
        default=2,
        min=1,
        max=4
    )
    
    # Object Properties
    bpy.types.Object.olat_optimizable = bpy.props.BoolProperty(
        name="OLAT Optimizable",
        description="Include this object in OLAT render",
        default=True
    )
    
    bpy.types.Object.olat_enabled = bpy.props.BoolProperty(
        name="OLAT Enabled",
        description="Include this object in any render pass",
        default=True
    )
    
    bpy.types.Object.olat_is_light = bpy.props.BoolProperty(
        name="OLAT Is Light",
        description="Identified as a light source",
        default=False
    )

    # World Properties
    bpy.types.World.olat_optimizable = bpy.props.BoolProperty(
        name="OLAT Optimizable",
        description="Include world in OLAT render",
        default=True
    )
    
    bpy.types.World.olat_enabled = bpy.props.BoolProperty(
        name="OLAT Enabled",
        description="Include world in any render pass",
        default=True
    )
    
    bpy.types.World.olat_is_light = bpy.props.BoolProperty(
        name="OLAT Is Light",
        description="Identified as a light source",
        default=False
    )

    # Collection Properties
    bpy.types.Collection.olat_expanded = bpy.props.BoolProperty(
        name="Expanded",
        default=True
    )
    bpy.types.Collection.olat_enabled = bpy.props.BoolProperty(
        name="Enabled",
        default=True,
        update=update_collection_enabled
    )
    bpy.types.Collection.olat_optimizable = bpy.props.BoolProperty(
        name="Optimizable",
        default=True,
        update=update_collection_optimizable
    )

def unregister():
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