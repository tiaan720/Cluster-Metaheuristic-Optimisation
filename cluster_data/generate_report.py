import os
import pandas as pd
import shutil
from fpdf import FPDF
from PIL import Image


# Step to delete the 'all_images' folder if it exists
def delete_folder_if_exists(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted existing folder: {folder_path}")


def combine_csv_files(input_folder, output_csv):
    combined_df = pd.DataFrame()

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file == "clustering_evaluation_results.csv":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                # Add a column to identify the source folder
                source_folder = os.path.basename(root)
                df["Source"] = source_folder
                # Extract the dataset name from the source folder name
                folder_parts = source_folder.split("_")
                dataset_name = "_".join(
                    folder_parts[:-3]
                )  # Exclude only the last part (algorithm)
                df["Dataset"] = dataset_name
                combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved to {output_csv}")


def combine_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = []

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                shutil.copy(file_path, output_folder)
                images.append(file_path)

    print(f"All images copied to {output_folder}")
    return images


def create_pdf_report(images, output_pdf):
    pdf = FPDF()
    for image_path in images:
        cover = Image.open(image_path)
        width, height = cover.size

        # Convert pixels to mm with 1 px = 0.264583 mm
        width, height = float(width * 0.264583), float(height * 0.264583)

        pdf.add_page()

        # A4 is 210mm x 297mm
        if width > height:
            pdf.image(image_path, 10, 10, 190)
        else:
            pdf.image(image_path, 10, 10, 0, 190)

    pdf.output(output_pdf, "F")
    print(f"PDF report saved to {output_pdf}")


def main():
    input_folder = "cluster_data"
    output_csv = "cluster_data/combined_clustering_results.csv"
    output_image_folder = "cluster_data/all_images"
    output_pdf = "cluster_data/clustering_report.pdf"

    # Delete the 'all_images' folder if it exists
    delete_folder_if_exists(output_image_folder)

    combine_csv_files(input_folder, output_csv)
    images = combine_images(input_folder, output_image_folder)
    create_pdf_report(images, output_pdf)


if __name__ == "__main__":
    main()
