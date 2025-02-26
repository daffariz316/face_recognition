const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultText = document.getElementById("result");

// Mengaktifkan kamera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error("Gagal mengakses kamera:", err));

function capture() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL("image/jpeg");

    fetch("/recognize", {
        method: "POST",
        body: JSON.stringify({ image: dataURL }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        if (data.faces.length > 0) {
            resultText.innerText = `Wajah dikenali: ${data.faces[0].name}`;
        } else {
            resultText.innerText = "Wajah tidak dikenali.";
        }
    })
    .catch(err => console.error("Error:", err));
}
