# -*- coding: utf-8 -*-
"""
Created on 2025/4/2 15:53
modified from: ws123_ortho_weedCAM_mapping.py

process on the overall orthomosaic tif image, use sliding window to process the image, and merge all Seg results
    Input: UAV orthomosaics.tif, area of interest (AOI.shp)
    Parms: win_size, step_size, out_dir
        1. read the orthomosaics.tif and AOI.shp, check the out_dir and create it if not exist
        2. use sliding window to load the area from orthomosaics.tif, check 1. valid area ratio>0.5, 2. is in AOI boundary
        3. predict the BL/GW segmentation masks
            save the Seg results tif to out_dir
        4. merge all Seg results to a single tif file use rasterio.merge for BL and GW
@author: LU
"""
import cv2
from tqdm import tqdm
from rasterio.windows import Window
from shapely.geometry import Point
from PIL import Image
import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
# Set up logging
import logging
import os
from datetime import datetime
from mmseg.apis import inference_model, init_model
from ws121_WM2_patch2weedCAM import show_cam_grad
from ws123_ortho_weedCAM_mapping import load_aoi_boundary, merge_all_BLGW_CAM_results
from ws127_infer_CAM2Seg_test import calca_zero_mask
import matplotlib.pyplot as plt

# %% Process orthomosaic image
def process_orthomosaic(ortho_tif_path, aoi_shp_path, results_out_dir,
                        win_size=1024, step_size=500, batch_size=32, Seg_mask_size=256,
                        use_log=True, mission_code='Ortho'):
    # Create output directories if they don't exist
    ori_image_dir = os.path.join(results_out_dir, 'ori_image/')
    BL_gray_dir = os.path.join(results_out_dir, 'BL_gray/')
    GW_gray_dir = os.path.join(results_out_dir, 'GW_gray/')
    Seg_view_dir = os.path.join(results_out_dir, 'Seg_view/')
    [os.makedirs(dir, exist_ok=True) for dir in [ori_image_dir, BL_gray_dir, GW_gray_dir, Seg_view_dir]]
    if use_log:
        # Create a unique log filename with timestamp
        log_filename = os.path.join(results_out_dir,
                                    f'orthomosaic_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also print to console
            ]
        )
        logger = logging.getLogger(__name__)
        # Log input parameters
        logger.info("Orthomosaic Processing Configuration:")
        logger.info(f"Orthomosaic Path: {ortho_tif_path}")
        logger.info(f"AOI Shapefile Path: {aoi_shp_path}")
        logger.info(f"Results Output Directory: {results_out_dir}")
        logger.info(f"Window Size: {win_size}")
        logger.info(f"Step Size: {step_size}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Seg Mask Size: {Seg_mask_size}")
        # note time
        start_time = datetime.now()
        logger.info(f"Start processing at {start_time}")

    # Load weed segmentation models
    prj_dir = r'F:\project_data\prj8_weedSeg\20250329_CAM2Seg_dataset\workdir\BiSeNetv2_FT_CAM2Seg_dp02_s1024_bs16x8_5k/'
    config_file = prj_dir + r'/BiSeNetv2_FT_CAM2Seg_dp02_s1024_bs16x8_5k.py'
    checkpoint_file = prj_dir + r'/best_mIoU_iter_4000.pth'
    seg_model = init_model(config_file, checkpoint_file, device='cuda:0')
    img_name_list = []
    # Load orthomosaic image to get its CRS
    with rasterio.open(ortho_tif_path) as src:
        ortho_crs = src.crs  # Get the CRS of the orthomosaic
        meta = src.meta.copy()
        width, height = src.width, src.height
        # Load AOI boundary and transform to ortho_crs
        aoi_polygon = load_aoi_boundary(aoi_shp_path, ortho_crs)

        # Sliding window processing with batch inference
        batch_img_list, batch_zero_mask, batch_windows = [], [], []
        for y in tqdm(range(0, height, step_size)):
            for x in range(0, width, step_size):
                if x + win_size > width or y + win_size > height:continue  # Skip out-of-bounds regions

                # Read window region from the orthomosaic
                window = Window(x, y, win_size, win_size)

                # Calculate window cebter in the entire image
                window_transform = src.window_transform(window)
                window_center = window_transform * (win_size // 2, win_size // 2)
                point = Point(window_center[0], window_center[1])
                # Check if the window center is within the AOI boundary
                if not aoi_polygon.contains(point):continue

                # Calculate valid pixel ratio
                # only keep the first 3 channels (RGB) for prediction
                #  DO NOT USE swapaxes !!! will cause flip error
                raster_array = src.read(window=window)
                img_array = np.stack([raster_array[0], raster_array[1], raster_array[2]], axis=2)  # test on RGB
                zero_mask = np.all(img_array == 0, axis=2)
                if np.count_nonzero(zero_mask) / zero_mask.size > 0.5:continue

                # save to ori_image_dir
                img = Image.fromarray(img_array)
                img_name = f'{mission_code}_x{x}_y{y}.jpg'
                img_name_list.append(img_name)
                img.save(os.path.join(ori_image_dir, img_name))
                batch_img_list.append(img_name)
                batch_zero_mask.append(zero_mask)
                batch_windows.append((x, y, window))

                # Process in batches
                if len(batch_img_list) == batch_size:
                    img_path_list = [os.path.join(ori_image_dir, img_name) for img_name in batch_img_list]
                    batch_results = inference_model(seg_model, img_path_list)
                    for i, img_name in enumerate(batch_img_list):
                        x, y, window = batch_windows[i]
                        img_path = os.path.join(ori_image_dir, img_name)
                        # Open the image
                        img = Image.open(img_path)
                        zero_mask = calca_zero_mask(np.array(img))
                        pred_mask_array = np.array(batch_results[i].pred_sem_seg.data[0].cpu()).astype(np.uint8) # Get the mask as a numpy array

                        pred_mask_array[zero_mask > 0] = 0
                        BL_seg_mask = (pred_mask_array==1).astype(np.uint8)*255
                        GW_seg_mask = (pred_mask_array==2).astype(np.uint8)*255

                        # Save Seg results for the current window
                        for seg_mask, seg_dir, weed_type in [(BL_seg_mask, BL_gray_dir, 'BL'), (GW_seg_mask, GW_gray_dir, 'GW')]:
                            # Define the output file paths
                            seg_tif_path = os.path.join(seg_dir, img_name.replace('.jpg', '.tif'))
                            seg_resize = cv2.resize(seg_mask, (Seg_mask_size, Seg_mask_size), interpolation=cv2.INTER_CUBIC)
                            # Update metadata for the current window
                            window_meta = meta.copy()
                            window_meta.update({
                                'width': Seg_mask_size,
                                'height': Seg_mask_size,
                                'transform': src.window_transform(window) * src.window_transform(window).scale(
                                    (win_size / Seg_mask_size),  # Scale x resolution
                                    (win_size / Seg_mask_size)  # Scale y resolution
                                ),
                                'count': 1,  # Single band
                                'dtype': 'uint8'  # Data type
                            })
                            # Save the CAM result as a GeoTIFF
                            with rasterio.open(seg_tif_path, 'w', **window_meta) as dst:
                                dst.write(seg_resize, 1)

                        cam_view_path = os.path.join(Seg_view_dir, img_name)
                        # Generate and save CAM visualization
                        show_cam_grad(pred_mask_array*110, img, title=img_name, out_path=cam_view_path)
                    # Reset batch
                    batch_img_list, batch_zero_mask, batch_windows = [], [], []
        # Process the remaining images in the batch
        if len(batch_img_list) > 0:
            img_path_list = [os.path.join(ori_image_dir, img_name) for img_name in batch_img_list]
            batch_results = inference_model(seg_model, img_path_list)
            for i, img_name in enumerate(batch_img_list):
                img_path = os.path.join(ori_image_dir, img_name)
                # Open the image
                img = Image.open(img_path)
                zero_mask = calca_zero_mask(np.array(img))
                pred_mask_array = np.array(batch_results[i].pred_sem_seg.data[0].cpu()).astype(np.uint8)
                pred_mask_array[zero_mask > 0] = 0
                BL_seg_mask = (pred_mask_array==1).astype(np.uint8)*255
                GW_seg_mask = (pred_mask_array==2).astype(np.uint8)*255
                for seg_mask, seg_dir, weed_type in [
                    (BL_seg_mask, BL_gray_dir, 'BL'),
                    (GW_seg_mask, GW_gray_dir, 'GW')]:
                    # Define the output file paths
                    cam_tif_path = os.path.join(seg_dir, img_name.replace('.jpg', '.tif'))
                    seg_resize = cv2.resize(seg_mask, (Seg_mask_size, Seg_mask_size), interpolation=cv2.INTER_CUBIC)

                    # Update metadata for the current window
                    window_meta = meta.copy()
                    window_meta.update({
                        'width': Seg_mask_size,
                        'height': Seg_mask_size,
                        'transform': src.window_transform(window) * src.window_transform(window).scale(
                            (win_size / Seg_mask_size),  # Scale x resolution
                            (win_size / Seg_mask_size)  # Scale y resolution
                        ),
                        'count': 1,  # Single band
                        'dtype': 'uint8'  # Data type
                    })
                    # Save the CAM result as a GeoTIFF
                    with rasterio.open(cam_tif_path, 'w', **window_meta) as dst:
                        dst.write(seg_resize, 1)
                    cam_view_path = os.path.join(Seg_view_dir, img_name)
                    # Generate and save CAM visualization
                    show_cam_grad(pred_mask_array*110, img, title=img_name, out_path=cam_view_path)
    print(f'Processing completed. Results saved in {results_out_dir}')
    if use_log:
        # note time
        end_time = datetime.now()
        logger.info(f"End processing at {end_time}")
        logger.info(f"Total processing time: {end_time - start_time}")
        # time per image
        logger.info(f"Time per image: {(end_time - start_time) / len(img_name_list)}")
        logger.info(f"Results saved in {results_out_dir}")
        logger.info(f"Log file saved in {log_filename}")
    return img_name_list


# %% Main function
if __name__ == '__main__':
    # 20250402 xszs7jc
    ortho_tif_path = r'F:\data\uav\20230816_xs_zhengshi7\20230816_xs_zhengshi7_P1_jc20\3_dsm_ortho\2_mosaic/20230816_xs_zhengshi7_P1_jc20_transparent_mosaic_group1.tif'
    aoi_shp_path = r'F:\project_data\prj8_weedSeg\20250312_WeedMap\AOI_area/B_test_area.shp'  # AOI shapefile
    results_out_dir = r'F:\project_data\prj8_weedSeg\20250402_orthoWeedSegMap/M20230816_xszs7jc_s500/'
    win_size, step_size = 1024, 500
    batch_size = 32
    Seg_mask_size = 64
    use_log = True
    mission_code = 'xszs7jc'
    process_orthomosaic(ortho_tif_path, aoi_shp_path, results_out_dir, win_size, step_size, batch_size,
                        Seg_mask_size=Seg_mask_size, use_log=True, mission_code=mission_code)
    # Merge CAM results
    merge_all_BLGW_CAM_results(results_out_dir, aoi_shp_path)

    # 20250402 on field A xszs5
    ortho_tif_path = r'F:\data\uav\20230717_xs_zhengshi5/20230717_xs_zhengshi5_P1_25_metashape_s_matched3.tif'
    aoi_shp_path = r'F:\project_data\prj8_weedSeg\20250312_WeedMap\AOI_area/A_train_area.shp'  # AOI shapefile
    results_out_dir = r'F:\project_data\prj8_weedSeg\20250402_orthoWeedSegMap/M20230717_xszs5_s500/'
    win_size, step_size = 1024, 500
    batch_size, Seg_mask_size = 24, 64
    mission_code = 'xszs5'
    process_orthomosaic(ortho_tif_path, aoi_shp_path, results_out_dir, win_size, step_size, batch_size,
                        Seg_mask_size=Seg_mask_size, use_log=True, mission_code=mission_code)
    merge_all_BLGW_CAM_results(results_out_dir, aoi_shp_path)
    # xszs6
    ortho_tif_path = r"F:\data\uav\20230731_xs_zhengshi6\20230731_xs_zhengshi6_P1_25\3_dsm_ortho\2_mosaic\20230731_xs_zhengshi6_P1_25_transparent_mosaic_group1.tif"
    aoi_shp_path = r'F:\project_data\prj8_weedSeg\20250312_WeedMap\AOI_area/A_train_area.shp'  # AOI shapefile
    results_out_dir = r'F:\project_data\prj8_weedSeg\20250402_orthoWeedSegMap/M20230731_xszs6_s500/'
    mission_code = 'xszs6'
    process_orthomosaic(ortho_tif_path, aoi_shp_path, results_out_dir, win_size, step_size, batch_size,
                        Seg_mask_size=Seg_mask_size, use_log=True, mission_code=mission_code)
    merge_all_BLGW_CAM_results(results_out_dir, aoi_shp_path)
    # xszs7
    ortho_tif_path = r'F:\data\uav\20230816_xs_zhengshi7/20230816_xs_zhengshi7_P1_25_metashape_matched.tif'
    aoi_shp_path = r'F:\project_data\prj8_weedSeg\20250312_WeedMap\AOI_area/A_train_area.shp'  # AOI shapefile
    results_out_dir = r'F:\project_data\prj8_weedSeg\20250402_orthoWeedSegMap/M20230816_xszs7_s500/'
    mission_code = 'xszs7'
    process_orthomosaic(ortho_tif_path, aoi_shp_path, results_out_dir, win_size, step_size, batch_size,
                        Seg_mask_size=Seg_mask_size, use_log=True, mission_code=mission_code)
    merge_all_BLGW_CAM_results(results_out_dir, aoi_shp_path)

