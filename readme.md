
#  SignAI: Generative AI for Personalized Sign Language Synthesis

## ✨ Project Description

The primary objective of this project is to develop a sophisticated **Artificial Intelligence system that generates realistic, personalized sign language videos directly from text or audio input**. The fundamental goal is to make digital information more accessible to Deaf and hard-of-hearing communities, bridging a critical communication gap in an increasingly digital world.

### Key Aims:

  * **End-to-End Pipeline:** Process spoken language (via Automatic Speech Recognition) or written text and translate it into a structured sign language format (**gloss**).
  * **Pose Generation:** Convert the gloss sequence into a sequence of accurate **2D skeletal poses**, capturing the complex kinematics of sign language.
  * **Accessibility:** Provide a proof-of-concept that can be applied in various domains, including education, media accessibility, workplace communication, and public service announcements.

-----

##  Architecture and Pipeline

This project is built around a multi-stage pipeline that converts natural language input into a visual sign animation:

1.  **Input:** Accepts plain **Text** (`--text`) or **Audio** (`--audio` / Streamlit mic/upload).
2.  **Speech-to-Text (STT):** Uses the **Whisper** model for audio transcription (in `retriever_frontend.py`).
3.  **Text-to-Gloss:** A **HuggingFace Seq2Seq Transformer Model** converts the transcribed text into a sequence of sign language glosses (e.g., "HELLO HOW ARE YOU").
4.  **Gloss Retrieval:** The system retrieves corresponding pose sequences from a pre-built **Gloss-Pose Library** (`gloss_pose_library_testval.pkl`).
5.  **Interpolation & Smoothing:** Transitions between different sign poses are smoothed using **linear interpolation** for fluid movement.
6.  **Video Synthesis:** **Matplotlib** is used with `FuncAnimation` and `FFMpegWriter` to render the 2D keypoint sequences into a final `.mp4` sign animation video.

-----

##  Repository Structure

```
.
├── retriever_frontend.py    # Main execution script for command-line use.
├── frontend.py             # Streamlit web application interface.
├── building_database.py    # Script to process How2Sign data and create the pose library.
├── gloss_pose_library_testval.pkl # (Large File) The pose database created by building_database.py.
└── best_model/             # (Large Files) Directory containing the Text-to-Gloss model.
```

-----
##  Visual Example

Below is a snapshot of the generated sign language animation, illustrating the real-time skeletal rendering of the translated gloss sequence.

![Example of Sign Language Animation Rendering](Figure1.png)

---
##  Setup and Installation

### Prerequisites

You will need **Python (3.8+)**, **Git**, and the **FFmpeg** library installed on your system to generate the video output.

### 1\. Clone the Repository

```bash
git clone https://github.com/Riyajhamb17/Generative-AI-for-personalized-Sign-Language-Synthesis.git
cd Generative-AI-for-personalized-Sign-Language-Synthesis
```

### 2\. Set up Virtual Environment

It is highly recommended to use a virtual environment:

```bash
python -m venv myenv
# On Windows
.\myenv\Scripts\activate
# On macOS/Linux
source myenv/bin/activate
```

### 3\. Install Dependencies

Install all necessary libraries:

```bash
pip install torch numpy matplotlib pandas tqdm transformers accelerate streamlit whisper-openai speechrecognition
```

### 4\. Required Data and Model Files

This project relies on two large files which must be present. Due to GitHub's file size limits, you must acquire these separately:

1.  **`best_model/model.safetensors`**: The trained **Text-to-Gloss** Transformer model.
2.  **`gloss_pose_library_testval.pkl`**: The pre-built database of poses.

These files must be placed in their respective locations as defined in the code.

-----

##  Usage

### Option 1: Command-Line Interface (CLI)

Use `retriever_frontend.py` to process text or audio directly:

**Text Input:**

```bash
python retriever_frontend.py --text "My name is John"
```

**Audio Input (e.g., MP3 or WAV file):**

```bash
python retriever_frontend.py --audio "/path/to/your/audio.mp3"
```

The resulting animation will be saved as `sign_output.mp4`.

### Option 2: Web Interface (Streamlit)

Run the Streamlit application for an interactive demo with text, file upload, and microphone input options:

```bash
streamlit run frontend.py
```

Open the provided URL in your browser to access the interface.

-----

## 💾 Building the Pose Database

The `gloss_pose_library_testval.pkl` file is created using the `building_database.py` script. This process requires the raw **How2Sign** dataset files.

1.  **Acquire Data:** Download the three How2Sign CSV metadata files and the corresponding **OpenPose 2D keypoints** folders (renamed to `keypoints/`).

2.  **Run Builder Script:**

    ```bash
    python building_database.py
    ```

The script will process the CSVs and JSON keypoint files, performing normalization and mapping frame sequences to their corresponding glosses, saving the result to `gloss_pose_library_testval.pkl`.