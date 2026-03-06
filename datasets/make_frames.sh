#!/bin/bash

# Check if necessary tools are installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it."
    exit 1
fi
if ! command -v ffprobe &> /dev/null; then
    echo "Error: ffprobe is not installed (part of ffmpeg). Please install ffmpeg."
    exit 1
fi
if ! command -v mogrify &> /dev/null; then
    echo "Error: mogrify is not installed (part of ImageMagick). Please install ImageMagick."
    exit 1
fi

# --- Configuration ---
OUTPUT_ROOT="frames"
TARGET_SIZE="512x512" # Target size after cropping and resizing

# --- Script Logic ---

# Create the root output directory if it doesn't exist
mkdir -p "$OUTPUT_ROOT"
echo "Ensured root output directory exists: $OUTPUT_ROOT"

# Loop through all video files provided as command line arguments
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <video1_path> [<video2_path> ...]"
    exit 1
fi

for input_video in "$@"; do
    if [ ! -f "$input_video" ]; then
        echo "Warning: Input file not found: $input_video. Skipping."
        continue
    fi

    # Derive a clean directory name from the video filename
    video_basename=$(basename -- "$input_video")
    video_name="${video_basename%.*}" # Remove file extension
    # Sanitize name to be filesystem friendly (optional but good practice)
    video_name=$(echo "$video_name" | sed 's/[^a-zA-Z0-9_-]/_/g')

    output_dir="$OUTPUT_ROOT/$video_name"

    # Create the specific output directory for this video
    mkdir -p "$output_dir"
    echo "--- Processing video: $input_video ---"
    echo "Output directory: $output_dir"

    # --- Step 1: Get original video dimensions using ffprobe ---
    dimensions=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$input_video")
    if [ -z "$dimensions" ]; then
        echo "Error: Could not get dimensions for $input_video. Skipping."
        continue
    fi
    IFS='x' read -r video_width video_height <<< "$dimensions"
    echo "Original dimensions: ${video_width}x${video_height}"

    # Calculate the size of the largest centered square
    if (( video_width < video_height )); then
        min_dim=$video_width
    else
        min_dim=$video_height
    fi
    echo "Largest centered square size: ${min_dim}x${min_dim}"

    # --- Step 2: Extract frames as JPGs (original size) ---
    # -start_number 1: start frame numbering at 1 (optional, but common)
    # -q:v 1: quality setting (ignored by jpg which is lossless)
    # -nostats -loglevel 0: suppress verbose ffmpeg output
    echo "Extracting frames..."
    ffmpeg -i "$input_video" -start_number 1 -q:v 1 \
      "$output_dir/frame_%06d.jpg" -nostats -loglevel 0
    if [ $? -ne 0 ]; then
        echo "Error: ffmpeg failed to extract frames from $input_video. Skipping mogrify."
        continue # Skip remaining steps for this video
    fi
    echo "Frame extraction complete."

    # --- Step 3: Crop to the largest centered square and resize ---
    # -gravity Center: sets the center as the reference point for cropping
    # -crop ${min_dim}x${min_dim}+0+0: crops a square of size min_dim x min_dim
    # -resize "$TARGET_SIZE!": resizes to the exact TARGET_SIZE, ignoring aspect ratio
    # "$output_dir"/*.jpg: applies the operations to all JPG files in the directory
    echo "Cropping and resizing frames to $TARGET_SIZE..."
    mogrify -gravity Center -crop ${min_dim}x${min_dim}+0+0 -resize "$TARGET_SIZE!" \
      "$output_dir"/*.jpg
    if [ $? -ne 0 ]; then
         echo "Error: mogrify failed to process frames in $output_dir."
         # Note: mogrify might exit with error if no files matched *.jpg,
         # but ffmpeg should have created them if it ran successfully.
    fi
    echo "Cropping and resizing complete."
    echo "Finished processing $input_video"
    echo "---"

done

echo "Dataset preparation script finished."
