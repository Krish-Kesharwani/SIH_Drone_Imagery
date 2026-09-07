import { useState } from "react";
import API from "../services/api";

function VideoUploader() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const uploadFile = async () => {
    if (!file) {
      setMessage("Select a video first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await API.post(
        "/upload-video",
        formData
      );

      setMessage(
        `Uploaded: ${response.data.filename}`
      );
    } catch (error) {
      setMessage("Upload failed");
      console.error(error);
    }
  };

  return (
    <div>
      <h2>Upload Drone Video</h2>

      <input
        type="file"
        accept="video/*"
        onChange={(e) =>
          setFile(e.target.files[0])
        }
      />

      <br />
      <br />

      <button onClick={uploadFile}>
        Upload
      </button>

      <p>{message}</p>
    </div>
  );
}

export default VideoUploader;