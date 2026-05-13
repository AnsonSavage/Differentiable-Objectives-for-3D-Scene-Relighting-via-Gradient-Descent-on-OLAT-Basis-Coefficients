import bpy
import os
from abc import ABC, abstractmethod

# === Settings ===
output_directory = "//rendered_lights"  # "//" makes it relative to the .blend file location
output_format = "OPEN_EXR"
base_filename = "light_pass_"

# Ensure output directory exists
full_output_path = bpy.path.abspath(output_directory)
os.makedirs(full_output_path, exist_ok=True)

def set_gpu():
    print("Attempting to configure GPU rendering...")
    try:
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "GPU"

        prefs = bpy.context.preferences.addons["cycles"].preferences

        # Try to set OPTIX
        try:
            prefs.compute_device_type = "OPTIX"
            prefs.get_devices() # This will error if 'OPTIX' is not a valid type
        except Exception as e:
            # If OPTIX fails, fall back to CUDA
            try:
                prefs.compute_device_type = "CUDA"
                prefs.get_devices()
            except Exception as e2:
                print(f"Warning: Could not set OPTIX or CUDA: {e2}")

        print(f"Set compute backend to: {prefs.compute_device_type}")

        # Call get_devices() again to populate the list for the chosen backend
        prefs.get_devices()

        # Enable all devices (GPU and CPU) following your example's structure
        # This iterates over the `preferences.devices` collection
        if not prefs.devices:
            raise Exception("No devices found for the selected backend.")

        print("Enabling devices...")
        for d in prefs.devices:
            d.use = True # Set device to be used
            print(f"Enabled: {d.name}, Type: {d.type}, Use: {d.use}")

    except Exception as e:
        print(f"Warning: Could not configure GPU rendering preferences: {e}")
        print("Will attempt to render with default scene settings.")

set_gpu()

class GetLightsStrategy(ABC):
    @abstractmethod
    def get_lights(self, scene):
        pass

class GetLightsByTypeStrategy(GetLightsStrategy):
    def get_lights(self, scene):
        return [obj for obj in scene.objects if obj.type == 'LIGHT']

class GetLightsByKeywordStrategy(GetLightsStrategy):
    def __init__(self, keyword):
        self.keyword = keyword

    def get_lights(self, scene):
        return [obj for obj in scene.objects if self.keyword in obj.name]

class GetMeshLightsStrategy(GetLightsStrategy):
    def get_lights(self, scene):
        lights = []
        for obj in scene.objects:
            if obj.type == 'MESH':
                is_light_object = False
                for slot in obj.material_slots:
                    mat = slot.material
                    if mat and mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            strength_input = None
                            color_input = None

                            # --- Case 1: Principled BSDF Shader ---
                            if node.type == 'BSDF_PRINCIPLED':
                                # Check for old ('Emission Strength') and new ('Emission') strength socket names
                                strength_input = node.inputs.get('Emission Strength') or node.inputs.get('Emission')
                                # Check for old ('Emission') and new ('Emission Color') color socket names
                                color_input = node.inputs.get('Emission') or node.inputs.get('Emission Color')

                            # --- Case 2: Emission Shader ---
                            elif node.type == 'EMISSION':
                                strength_input = node.inputs.get('Strength')
                                color_input = node.inputs.get('Color')

                            # --- Process the found node ---
                            if strength_input and color_input:
                                # Condition 1: Emission strength is > 0 or has a node connected.
                                has_strength = strength_input.is_linked or strength_input.default_value > 0
                                
                                # Condition 2: Color is not black or has a node connected.
                                # any(color[:3]) checks if any of R, G, B channels are non-zero.
                                is_not_black = color_input.is_linked or any(color_input.default_value[:3])

                                if has_strength and is_not_black:
                                    lights.append(obj)
                                    is_light_object = True
                                    break  # Exit node loop, we've confirmed this object is a light
                    if is_light_object:
                        break  # Exit material slot loop and move to the next object
        return lights

# --- UNCHANGED CLASSES ---
class CustomLightListStrategy(GetLightsStrategy):
    def get_lights(self, scene):
        explicit_lights = GetLightsByTypeStrategy().get_lights(scene)
        explicit_lights = [light for light in explicit_lights if 'portal' not in light.name.lower()]  # Remove portal lights
        mesh_lights = GetMeshLightsStrategy().get_lights(scene)
        lights = explicit_lights + mesh_lights
        return lights

# Get all light objects in the scene
# lights = AllLightsByLightTypeStrategy().get_lights(bpy.context.scene)
# light_keyword = 'LGT'
# lights = GetLightsByKeywordStrategy(light_keyword).get_lights(bpy.context.scene)
lights = CustomLightListStrategy().get_lights(bpy.context.scene)


# Save original visibility states
original_visibility = {light.name: light.hide_render for light in lights}

# Store original render settings
original_filepath = bpy.context.scene.render.filepath
original_format = bpy.context.scene.render.image_settings.file_format

# Set render settings
bpy.context.scene.render.image_settings.file_format = output_format

for i, active_light in enumerate(lights):
    print(f"Rendering pass {i + 1}/{len(lights)}: {active_light.name}")

    # Disable all lights
    for light in lights:
        light.hide_render = True

    # Enable only the current light
    active_light.hide_render = False

    # Set the output file path
    filename = f"{base_filename}{i:03d}_{active_light.name}.exr"
    bpy.context.scene.render.filepath = os.path.join(full_output_path, filename)

    # Render and save the image
    bpy.ops.render.render(write_still=True)

# Restore original light visibility
for light in lights:
    light.hide_render = original_visibility[light.name]

# Restore render settings
bpy.context.scene.render.filepath = original_filepath
bpy.context.scene.render.image_settings.file_format = original_format

print("Done rendering all individual light passes.")
