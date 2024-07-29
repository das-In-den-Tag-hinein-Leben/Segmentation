import os
import subprocess
import nibabel as nib
from config.settings import path_routing

def get_nifti_meta(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return img, data
    except Exception as e:
        print(f"Ошибка при загрузке файла NIfTI: {e}")
        return None, None


def fastsurfer(t1_image_paths, num_research):
    lic_data = path_routing.lic_path

    for t1_image_path in t1_image_paths:
        print(f"Проверка наличия файла T1 по пути: {t1_image_path}")
        if not os.path.isfile(t1_image_path):
            print("Error: T1 image file not found at", t1_image_path)
            continue

        # Generate unique SUBJECT ID based on filename
        filename = os.path.basename(t1_image_path)
        subject_id = os.path.splitext(filename)[0]

        # Replace backslashes with forward slashes for Docker compatibility
        t1_image_path_docker = t1_image_path.replace('\\', '/')
        input_dir_docker = os.path.dirname(t1_image_path).replace('\\', '/')
        lic_data_docker = lic_data.replace('\\', '/')

        # Ensure the SUBJECT ID is Docker-friendly (no spaces)
        docker_subject_id = num_research.replace(' ', '_')

        # Run the FastSurfer script
        print(f"Running the FastSurfer detached container for {num_research}")
        command = [
            'docker', 'run', '--gpus', 'all',
            '-v', f"{input_dir_docker}:/data",
            '-v', f"{input_dir_docker}:/output",
            '-v', f"{lic_data_docker}:/fs_license/license.txt",
            'deepmi/fastsurfer:gpu-v2.1.2',
            '--allow_root',
            '--fs_license', '/fs_license/license.txt',
            '--t1', f"/data/{filename}",
            '--device', 'cuda',
            '--sid', docker_subject_id,
            '--sd', '/output',
            '--parallel'
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error running FastSurfer script:", e)

        return docker_subject_id



# Example usage
# fastsurfer([r"E:\1\Segmentation\database\for_test\research_207494001 OLE W 94\NIFTI\801_t1w_post_gad.nii.gz"], "1")
