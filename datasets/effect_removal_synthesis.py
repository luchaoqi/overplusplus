import logging
from absl import app
from absl import flags
import uuid
import os
import glob
import gc
from typing import Optional, Union, Tuple
import sys
import bpy
sys.path.append('./')
import numpy as np
import mediapy as media
import traitlets as tl
import kubric as kb
from kubric.core import traits as ktl
from kubric.renderer.blender import Blender as KubricBlender
from kubric.renderer.blender import AttributeSetter
from kubric.renderer.blender import KeyframeSetter
from kubric.simulator.pybullet import PyBullet as KubricSimulator


logging.basicConfig(level='INFO')


_OUTPUT_DIR = flags.DEFINE_string(
	'output_dir',
	'test_outputs',
	'output root folder'
)

_VIDEO_LENGTH = flags.DEFINE_integer(
	'video_length',
	80,
	'length of generated video'
)

_RENDER_WIDTH = flags.DEFINE_integer(
	'render_width', 256,
	'the resolution to render'
)

_RANGE_MATERIAL_METALLIC = flags.DEFINE_list(
	'range_material_metallic', [0.8, 1.0],
	'random range of metallic attribute'
)

_RANGE_MATERIAL_SPECULAR = flags.DEFINE_list(
	'range_material_specular', [0.0, 1.0],
	'random range of specular attribute'
)

_RANGE_MATERIAL_SPECULAR_TINT = flags.DEFINE_list(
	'range_material_specular_tint', [0.0, 1.0],
	'random range of specular-tint attribute'
)

_RANGE_MATERIAL_roughness = flags.DEFINE_list(
	'range_material_roughness', [0.0, 0.6],
	'random range of roughness attribute'
)

_RANGE_MATERIAL_IOR = flags.DEFINE_list(
	'range_material_ior', [1.0, 2.5],
	'random range of IOR attribute'
)

_RANGE_MATERIAL_TRANSMISSION = flags.DEFINE_list(
	'range_material_transmission', [0.0, 1.0],
	'random range of transmission attribute'
)

_RANGE_MATERIAL_TRANSMISSION_ROUGHNESS = flags.DEFINE_list(
	'range_material_transmission_roughness', [0.0, 0.5],
	'random range of transmission-roughness attribute'
)

_RANGE_MATERIAL_ALPHA = flags.DEFINE_list(
	'range_material_alpha', [0.7, 1.0],
	'random range of alpha attribute'
)

_RANGE_OBJECT_SCALE = flags.DEFINE_list(
	'range_object_scale', [1.0, 3.0], # [0.5, 1.5],
	'random range of object scale'
)

_RANGE_NUM_OBJS = flags.DEFINE_list(
	'range_num_objs', [1, 6],
	'random range of number of objects, high val is exclusive'
)

_SEED = flags.DEFINE_integer(
	'seed', 0,
	'random seed'
)

_CAMERA_MAX_OFFSET = flags.DEFINE_float(
	'camera_max_offset', 1.0,
	'max range of camera position offset from origin (0, 0, 0)'
)

_CAMERA_RANGE_POSITION = flags.DEFINE_list(
	'camera_range_position', [5.0, 15.0],
	'the range of a sphere shell for camera position'
)

_CAMERA_RANGE_LOOKAT = flags.DEFINE_list(
	'camera_range_lookat', [0.0, 2.5],
	'the range of a sphere shell for camera lookat location'
)

_CAMERA_MAX_MOVEMENT = flags.DEFINE_float(
	'camera_max_movement', 4.0,
	'max range of camera movement between start and end'
)

_CAMERA_RANGE_FOCAL = flags.DEFINE_list(
	'camera_range_focal', [35, 45],
	'range of camera focal with sensor width = 32'
)

_LIGHT_RANGE_POSITION = flags.DEFINE_list(
	'light_range_position', [0, 4],
	'range of lighting source position'
)

_LIGHT_RANGE_LOOKAT = flags.DEFINE_list(
	'light_range_lookat', [0, 1],
	'range of lighting source lookat location'
)

_LIGHT_RANGE_INTENSITY = flags.DEFINE_list(
	'light_range_intensity', [4, 6],
	'range of lighting intensity'
)

_PROB_WALL = flags.DEFINE_float(
	'prob_wall', 0.25,
	'probability of generating walls'
)

_PROB_GROUND_MIRROR = flags.DEFINE_float(
	'prob_ground_mirror', 0.5,
	'probability of generating mirror on the ground'
)

_PROB_SAME_OBJECT = flags.DEFINE_float(
	'prob_same_object', 0.5,
	'probability of generating same multiple objects'
)

_PROB_STATIONARY_OBJECT = flags.DEFINE_float(
	'prob_stationary_object', 0.5,
	'probability of generation stationary objects'
)

_PROB_EXTREME_LIGHTING = flags.DEFINE_float(
	'prob_extreme_lighting', 0.5,
	'probability of extremely low angle lighting'
)

_MASK_MODE = flags.DEFINE_string(
	'mask_mode', 'trimask',
	'binary or trimask'
)

_SAVE_IMAGES = flags.DEFINE_boolean(
	'save_images', False,
	'whether save images or not'
)

_SAVE_STATE = flags.DEFINE_boolean(
	'save_state', False,
	'whether save the .blend file or not'
)

_DEBUG_USE_SIMPLE_OBJECT = False
SPAWN_REGION = [(-4, -4, 2.), (4, 4, 4.)]
STATIONARY_SPAWN_REGION = [(-4, -4, 1.), (4, 4, 2.)]
VELOCITY_RANGE = [(-3., -3., -3.), (3., 3., 3.)]


# blender assets
material_transparent = kb.TransparentMaterial(alpha=0.0)
kubasic_source = kb.AssetSource.from_manifest('gs://kubric-public/assets/KuBasic/KuBasic.json')
hdri_source = kb.AssetSource.from_manifest('gs://kubric-public/assets/HDRI_haven/HDRI_haven.json')
gso_source = kb.AssetSource.from_manifest('gs://kubric-public/assets/GSO/GSO.json')


def setup_scene(rng):
	scene = kb.Scene(resolution=(_RENDER_WIDTH.value, _RENDER_WIDTH.value))
	scene.frame_end = _VIDEO_LENGTH.value
	scene.frame_rate = int(rng.choice([8, 12, 16, 20, 24]))
	scene.step_rate = int(rng.integers(8, 12) * scene.frame_rate)
	return scene


def get_random_pbsdf_material(rng):
	# https://docs.blender.org/manual/en/2.80/render/shader_nodes/shader/principled.html
	rand = rng.uniform
	material = kb.PrincipledBSDFMaterial(
		color=kb.random_hue_color(rng=rng),
		specular=rand(*_RANGE_MATERIAL_SPECULAR.value),
		specular_tint=rand(*_RANGE_MATERIAL_SPECULAR_TINT.value),
		roughness=rand(*_RANGE_MATERIAL_roughness.value),
		transmission=rand(*_RANGE_MATERIAL_TRANSMISSION.value),
		transmission_roughness=rand(*_RANGE_MATERIAL_TRANSMISSION_ROUGHNESS.value),
		ior=rand(*_RANGE_MATERIAL_IOR.value),
		metallic=rand(*_RANGE_MATERIAL_METALLIC.value),
		alpha=rand(*_RANGE_MATERIAL_ALPHA.value),
	)
	return material


def set_principled_bsdf_attributes(mat_pbsdf, pbsdf_node):
	attributes = [
		('Specular', 'specular'),
		('Roughness', 'roughness'),
		('Metallic', 'metallic'),
		('Specular Tint', 'specular_tint'),
		('IOR', 'ior'),
		('Transmission', 'transmission'),
		('Transmission Roughness', 'transmission_roughness'),
		('Emission', 'emission'),
	]
	for attr_bpy, attr in attributes:
		pbsdf_node.inputs[attr_bpy].default_value = (getattr(mat_pbsdf, attr))



def mix_texture_and_pbsdf(obj_blender, mat_pbsdf, texture_filename=None):
	mat = obj_blender.data.materials[0]
	mat.use_nodes = True
	tree = mat.node_tree
	texture_node = tree.nodes['Image Texture']
	pbsdf_node = tree.nodes['Principled BSDF']
	output_node = tree.nodes['Material Output']

	if texture_filename:
		texture_node.image = bpy.data.images.load(texture_filename)

	set_principled_bsdf_attributes(mat_pbsdf, pbsdf_node)


def generate_background_asset(scene, renderer, rng):
	bg_hdri = hdri_source.create(asset_id=rng.choice(hdri_source.all_asset_ids))
	renderer._set_ambient_light_hdri(bg_hdri.filename)

	dome = kubasic_source.create(
		asset_id='dome', name='dome', static=True, background=True
	)
	assert isinstance(dome, kb.FileBasedObject)
	scene += dome
	dome_blender = dome.linked_objects[renderer]

	mat_pbsdf = get_random_pbsdf_material(rng)

	mix_texture_and_pbsdf(dome_blender, mat_pbsdf, bg_hdri.filename)
	return 1


def generate_ground_mirror(scene, renderer, rng):
	color = kb.random_hue_color(rng=rng)
	material = get_random_pbsdf_material(rng)
	material.metallic = rng.uniform(2.4, 3.0)
	material.transmission = rng.uniform(0., 0.1)
	material.roughness = 0.
	material.transmission_roughness = 0.
	mirror_size = rng.uniform(4.0, 8.0)
	position = (rng.uniform(-2., 2.), rng.uniform(-2., 2.), 0.)
	scene += kb.Cube(
		name='ground_mirror', scale=(mirror_size, mirror_size, 0.1),
		position=(0, 0, 0.1),
		static=True,
		material=material
	)
	return 1


def generate_wall(
		scene, renderer,
		camera_position, camera_lookat,
		light_position, light_lookat,
		rng=np.random.default_rng(),
	):
	color = kb.random_hue_color(rng=rng)
	material = get_random_pbsdf_material(rng)
	camera_position = np.array(camera_position)
	camera_lookat = np.array(camera_lookat)
	light_position = np.array(light_position)
	light_lookat = np.array(light_lookat)
	wall_position = 1.5 * (camera_lookat - camera_position) + camera_position
	# perturb
	wall_position[0] += rng.uniform(-1.0, 1.0)
	wall_position[1] += rng.uniform(-1.0, 1.0)
	camera_position[0] += rng.uniform(-1.0, 1.0)
	camera_position[1] += rng.uniform(-1.0, 1.0)

	lookat = camera_position - wall_position
	wall_size = rng.uniform(2.0, 3.0)
	wall_position[2] = wall_size * 0.5
	lookat[2] = wall_size * 0.5
	scene += kb.Cube(
		name='wall', scale=(wall_size, wall_size, 0.1), position=wall_position,
		look_at=lookat,
		static=True,
		material=get_random_pbsdf_material(rng),
	)
	return 1


def generate_environment_cube(scene, renderer, rng):
	scene += kb.Cube(
		name='floor', scale=(3, 3, 0.1), position=(0, 0, -0.1), static=True,
		material=get_random_pbsdf_material(rng),
	)
	return 1


def generate_environment(scene, renderer, rng):
	obj_count = 0
	if _DEBUG_USE_SIMPLE_OBJECT:
		obj_count += generate_environment_cube(scene, renderer, rng)
	else:
		obj_count += generate_background_asset(scene, renderer, rng)

	num_objs, light_pos, light_lookat = generate_light(scene, renderer, rng)
	obj_count += num_objs

	camera_pos, camera_lookat = generate_camera(scene, renderer, rng)
	obj_count += 1

	if rng.uniform(0, 1) < _PROB_WALL.value:
		obj_count += generate_wall(
			scene, renderer, camera_pos, camera_lookat,
			light_pos, light_lookat, rng,
		)
		
	if rng.uniform(0, 1) < _PROB_GROUND_MIRROR.value:
		obj_count += generate_ground_mirror(
			scene, renderer, rng,
		)
	return obj_count

def get_linear_start_end(
		move_speed, inner_radius=4., outer_radius=10., z_offset=0.1,
		rng=np.random.default_rng()
	):
	while True:
		start = np.array(kb.sample_point_in_half_sphere_shell(
			inner_radius, outer_radius, z_offset, rng
		))
		if move_speed is not None:
			direction = rng.normal(size=3) - 0.5
			movement = direction / np.linalg.norm(direction) * move_speed
			end = start + movement
			if (inner_radius <= np.linalg.norm(end) <= outer_radius and end[2] > z_offset):
				break
		else:
			end = np.array(kb.sample_point_in_half_sphere_shell(
				inner_radius, outer_radius, z_offset, rng
			))
			break
	return start, end


def generate_light(scene, renderer, rng):
	use_extreme_lighting = rng.uniform(0, 1) < _PROB_EXTREME_LIGHTING.value
	while True:
		position = np.array(kb.sample_point_in_half_sphere_shell(
			_LIGHT_RANGE_POSITION.value[0],
			_LIGHT_RANGE_POSITION.value[1],
			0.0, rng
		))

		lookat = np.array(kb.sample_point_in_half_sphere_shell(
			_LIGHT_RANGE_LOOKAT.value[0],
			_LIGHT_RANGE_LOOKAT.value[1],
			0.0, rng
		))
		if not use_extreme_lighting: break
		if position[2] < 0.2: break

	intensity = rng.uniform(*_LIGHT_RANGE_INTENSITY.value)

	scene += kb.DirectionalLight(
		name='sun', position=position, look_at=lookat, intensity=intensity
	)
	return 1, position, lookat


def generate_camera(scene, renderer, rng):
	pos_start_offset, pos_end_offset = get_linear_start_end(
		move_speed=None,
		inner_radius=0.,
		outer_radius=_CAMERA_MAX_OFFSET.value,
		z_offset=0.0,
		rng=rng,
	)
	
	pos_start, pos_end = get_linear_start_end(
		move_speed=rng.uniform(low=0., high=_CAMERA_MAX_MOVEMENT.value),
		inner_radius=_CAMERA_RANGE_POSITION.value[0],
		outer_radius=_CAMERA_RANGE_POSITION.value[1],
		z_offset=0.1,
		rng=rng,
	)
	pos_start += pos_start_offset
	pos_end += pos_end_offset

	lookat_start, lookat_end = get_linear_start_end(
		move_speed=None,
		inner_radius=_CAMERA_RANGE_LOOKAT.value[0],
		outer_radius=_CAMERA_RANGE_LOOKAT.value[1],
		z_offset=0.0,
		rng=rng,
	)
	
	focal = rng.uniform(*_CAMERA_RANGE_FOCAL.value)

	scene.camera = kb.PerspectiveCamera(focal_length=focal, sensor_width=32)
	
	length = scene.frame_end
	for t in range(length):
		interp = (t / length)
		scene.camera.position = (
			np.array(pos_start) * interp + np.array(pos_end) * (1 - interp)
		)
		scene.camera.look_at(
			np.array(lookat_start) * interp + np.array(lookat_end) * (1 - interp)
		)
		scene.camera.keyframe_insert('position', t)
		scene.camera.keyframe_insert('quaternion', t)

	return (pos_start + pos_end) * 0.5, (lookat_start + lookat_end) * 0.5


def generate_object_simple(scene, renderer, simulator, rng):
	velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
	color = kb.random_hue_color(rng=rng)
	material = get_random_pbsdf_material(rng)
	obj = kb.Sphere(scale=0.15, velocity=velocity, material=material)
	scene += obj
	kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
	return obj


def generate_object(
		scene, renderer, simulator, rng, asset_id=None, spawn_region=None
	):
	if _DEBUG_USE_SIMPLE_OBJECT:
		return generate_object_simple(scene, renderer, simulator, rng)
	
	if asset_id is None:
		asset_id = rng.choice(gso_source.all_asset_ids)
	
	if spawn_region is None:
		spawn_region = SPAWN_REGION

	scale = rng.uniform(*_RANGE_OBJECT_SCALE.value)	
	obj = gso_source.create(asset_id=asset_id)
	assert isinstance(obj, kb.FileBasedObject)
	obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
	try:
		kb.move_until_no_overlap(
			obj, simulator, spawn_region=spawn_region, rng=rng)
	except:
		pass

	scene += obj
	obj_blender = obj.linked_objects[renderer]
	mat_pbsdf = get_random_pbsdf_material(rng)
	mix_texture_and_pbsdf(obj_blender, mat_pbsdf)
	obj.velocity = (
		rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0]
	)
	return obj


def render_full(scene, renderer):
	return renderer.render(return_layers=('rgba',))


def get_mask_from_segmentation(segmentation, num_background_objs, remove_ids):
	mask = np.ones_like(segmentation).astype(float)
	start_id = num_background_objs + 1
	for i in remove_ids:
		mask = np.where(segmentation == (start_id + i), 0., mask)
	
	if _MASK_MODE.value == 'trimask':
		mask = np.where(segmentation < start_id, 128. / 255., mask)  # background elements

	return mask


def render_removed(
		scene, renderer, obj_list, remove_ids, num_background_objs
	):
	def _render(material):
		start_id = num_background_objs
		rm_names = []
		for i in remove_ids:
			obj_list[i].material = material
			rm_names.append(obj_list[i].name)
		return renderer.render(
			return_layers=('rgba', 'segmentation'),
			transparent_obj_names=rm_names,
		)
	rendered_removed = _render(material_transparent)

	mask_removed = get_mask_from_segmentation(
		rendered_removed['segmentation'], num_background_objs, remove_ids
	)
	return {
		'rgba': rendered_removed['rgba'].astype(np.uint8),
		'mask': mask_removed,
	}


def save_images(output_dir, rendered_full, rendered_removed):
	output_rgb_full_dir = os.path.join(output_dir, 'rgb_full')
	output_rgb_rm_dir = os.path.join(output_dir, 'rgb_removed')
	output_mask_dir = os.path.join(output_dir, 'mask')
	output_tuple_dir = os.path.join(output_dir, 'tuple')
	
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(output_rgb_full_dir, exist_ok=True)
	os.makedirs(output_rgb_rm_dir, exist_ok=True)
	os.makedirs(output_mask_dir, exist_ok=True)
	os.makedirs(output_tuple_dir, exist_ok=True)
	
	rgb_full = rendered_full['rgba'][..., :3].astype(np.uint8)
	rgb_rm = rendered_removed['rgba'][..., :3].astype(np.uint8)
	mask = np.repeat(rendered_removed['mask'] * 255, 3, axis=-1).astype(np.uint8)

	kb.file_io.write_rgb_batch(
		rgb_full, output_rgb_full_dir, file_template='{:05d}.png'
	)
	kb.file_io.write_rgb_batch(
		rgb_rm, output_rgb_rm_dir, file_template='{:05d}.png'
	)
	kb.file_io.write_rgb_batch(
		mask, output_mask_dir, file_template='{:05d}.png'
	)
	kb.file_io.write_rgb_batch(
		np.concatenate((rgb_full, mask, rgb_rm), axis=-2),
		output_tuple_dir, file_template='{:05d}.png'
	)


def save_videos(output_dir, rendered_full, rendered_removed):
	rgb_full = rendered_full['rgba'][..., :3].astype(np.uint8)
	rgb_rm = rendered_removed['rgba'][..., :3].astype(np.uint8)
	mask = np.repeat(rendered_removed['mask'] * 255, 3, axis=-1)
	mask = np.uint8(mask)

	os.makedirs(output_dir, exist_ok=True)
	media.write_video(
		os.path.join(output_dir, 'rgb_full.mp4'), rgb_full, fps=16
	)
	media.write_video(
		os.path.join(output_dir, 'rgb_removed.mp4'), rgb_rm, fps=16
	)
	media.write_video(
		os.path.join(output_dir, 'mask.mp4'), mask, fps=16
	)
	media.write_video(
		os.path.join(output_dir, 'tuple.mp4'),
		np.concatenate((rgb_full, mask, rgb_rm), axis=-2),
		fps=16
	)

def generate_tuple(output_dir, seed=0):
	rng = np.random.default_rng(seed)
	scene = setup_scene(rng)
	renderer = KubricBlender(scene, background_transparency=True)
	simulator = KubricSimulator(scene)
	
	# background
	num_background_objs = generate_environment(scene, renderer, rng)

	# object
	num_objs = rng.integers(*_RANGE_NUM_OBJS.value)
	num_objs_to_remove = rng.integers(1, num_objs + 1)
	remove_ids = rng.choice(
		list(range(num_objs)), num_objs_to_remove
	).tolist()

	is_stationary_object = (
		rng.uniform(0, 1) < _PROB_STATIONARY_OBJECT.value
	)
	if is_stationary_object:
		spawn_region = STATIONARY_SPAWN_REGION
	else:
		spawn_region = SPAWN_REGION

	if rng.uniform(0, 1) < _PROB_SAME_OBJECT.value:	
		asset_id = rng.choice(gso_source.all_asset_ids)
	else:
		asset_id = None
	obj_list = []
	for i in range(num_objs):
		obj = generate_object(
			scene, renderer, simulator, rng,
			asset_id=asset_id, spawn_region=spawn_region
		)
		obj_list.append(obj)

	if not is_stationary_object:
		simulator.run()

	rendered_full = render_full(scene, renderer)
	rendered_removed = render_removed(
		scene, renderer, obj_list, remove_ids, num_background_objs)

	# output
	save_videos(output_dir, rendered_full, rendered_removed)
	if _SAVE_IMAGES.value:
		save_images(output_dir, rendered_full, rendered_removed)

	# save blender file
	if _SAVE_STATE.value:
		renderer.save_state(os.path.join(output_dir, 'simulator.blend'))


def main(_):
	root_output_dir = _OUTPUT_DIR.value
	generate_tuple(root_output_dir, seed=_SEED.value)


if __name__ == '__main__':
	app.run(main)
