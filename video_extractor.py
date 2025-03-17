import cv2
import os
import logging
import math
from pathlib import Path


def setup_logging():
    """Configure logging with appropriate format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def split_video_into_snippets(input_path, output_dir, snippet_duration=2,
                              fps=8):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting to process video: {input_path}")
    logger.info(f"Output directory: {output_dir}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open the video
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        logger.error(f"Failed to open video file: {input_path}")
        return

    video_fps = video.get(cv2.CAP_PROP_FPS)  # Original FPS
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps  # Duration in seconds
    logger.info(
        f"Video properties - FPS: {video_fps}, Total frames: {total_frames}, "
        f"Duration: {duration:.2f} seconds")

    # Video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video dimensions: {width}x{height}")

    # Better approach: First collect all frame indices at target FPS
    # This ensures even sampling throughout the video
    source_timestamps = [i/video_fps for i in range(total_frames)]
    target_timestamps = [i/fps for i in range(int(duration*fps) + 1)]

    # For each target timestamp, find the closest source frame
    frame_indices = []
    for target_time in target_timestamps:
        # Find closest frame index
        closest_idx = min(range(len(source_timestamps)),
                          key=lambda i: abs(source_timestamps[i] - target_time))
        frame_indices.append(closest_idx)

    logger.info(f"Generated {len(frame_indices)} frame indices for sampling")

    # Calculate parameters
    frames_per_snippet = snippet_duration * fps
    total_snippets = math.ceil(len(frame_indices) / frames_per_snippet)

    logger.info(f"Will extract {total_snippets} snippets at {fps} FPS")

    # Codec selection for different platforms
    if os.name == 'posix' and os.uname().sysname == 'Darwin':
        # H.264 codec for macOS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        file_ext = '.mp4'
    else:
        # XVID codec for Windows/Linux
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        file_ext = '.avi'

    # Process the video by reading frames at the identified indices
    snippet_num = 0
    frames_processed = 0

    # Split frame indices into snippet-sized chunks
    for chunk_start in range(0, len(frame_indices), frames_per_snippet):
        chunk_indices = frame_indices[chunk_start:chunk_start +
                                      frames_per_snippet]
        if not chunk_indices:
            break

        logger.info(
            f"Processing snippet {snippet_num} with {len(chunk_indices)} frames", end="\r")

        # Create output file
        output_file = os.path.join(
            output_dir, f"snippet_{snippet_num:04d}{file_ext}")

        # Read frames for this snippet
        snippet_frames = []
        for frame_idx in chunk_indices:
            # Set position and read frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            if not ret:
                logger.warning(f"Failed to read frame at index {frame_idx}")
                continue
            snippet_frames.append(frame)

        if snippet_frames:
            # Create writer with the exact FPS we want
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error(f"Failed to create output file: {output_file}")
                # Try backup codec
                backup_output = os.path.join(
                    output_dir, f"snippet_{snippet_num:04d}.avi")
                backup_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                logger.info(f"Trying backup codec with {backup_output}")
                out = cv2.VideoWriter(
                    backup_output, backup_fourcc, fps, (width, height))

                if not out.isOpened():
                    logger.error("Backup codec also failed, skipping")
                    snippet_num += 1
                    continue

            # Write all frames in this snippet
            for frame in snippet_frames:
                out.write(frame)

            frames_processed += len(snippet_frames)
            out.release()
            logger.info(
                f"Completed snippet {snippet_num} with {len(snippet_frames)} frames", end="\r")
        else:
            logger.warning(
                f"No frames collected for snippet {snippet_num}, skipping")

        snippet_num += 1

    video.release()
    logger.info(
        f"Completed processing {input_path}. Created {snippet_num} snippets "
        f"with {frames_processed} total frames.")


def process_videos_in_directory(input_dir, output_base_dir):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch processing from directory: {input_dir}")

    video_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_count += 1
                input_path = os.path.join(root, file)
                # Create output subdir based on input file name
                output_dir = os.path.join(output_base_dir, Path(file).stem)
                logger.info(f"Processing video {video_count}: {file}")
                split_video_into_snippets(input_path, output_dir)

    logger.info(f"Batch processing complete. Processed {video_count} videos.")


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()

    # Replace with your input folder
    input_directory = (
        "training_data/input"
    )
    # Replace with your output folder
    output_base_directory = "training_data/output"

    logger.info("Starting video extraction script")
    try:
        process_videos_in_directory(input_directory, output_base_directory)
    except Exception as e:
        logger.error(
            f"An error occurred during processing: {str(e)}", exc_info=True)
    logger.info("Script execution completed")
