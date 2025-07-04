#!/bin/bash

BASE_URL="https://zenodo.org/record/<your-record-number>/files"

# Define file groups
annotations_files=("annotations.csv" "candidates.csv" "evaluationScript.zip" "sampleSubmission.csv")
masks_files=("seg-lungs-LUNA16.zip")

# Generate subset filenames subset0.zip to subset6.zip
subsets_files=()
for i in {0..6}; do
    subsets_files+=("subset${i}.zip")
done

# Generic function to download, unzip and clean up zip files
download_files() {
    local folder="$1"
    shift
    local files=("$@")

    # Create target folder if it doesn't exist
    if [ ! -d "$folder" ]; then
        echo "Creating folder: $folder"
        mkdir -p "$folder"
    else
        echo "Folder already exists: $folder"
    fi

    # Iterate over each file
    for file in "${files[@]}"; do
        local filepath="$folder/$file"
        echo "Downloading $file to $folder/"
        wget -c -P "$folder" "$BASE_URL/$file"

        # If file is a zip, unzip and delete it
        if [[ "$file" == *.zip ]]; then
            echo "Unzipping $file..."
            unzip -q "$filepath" -d "$folder" && rm -f "$filepath"
            echo "Extracted and removed $file"
        fi
    done
}

download_files "annotations" "${annotations_files[@]}"
download_files "masks" "${masks_files[@]}"
download_files "subsets" "${subsets_files[@]}"

echo "All files downloaded, extracted, and cleaned up successfully."