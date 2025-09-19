document.getElementById('uploadBtn').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    const output = document.getElementById('output');

    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            output.innerHTML = ''; // Clear previous output
            output.appendChild(img); // Display uploaded image
        };
        reader.readAsDataURL(fileInput.files[0]);
    } else {
        alert('Please select an image to upload.');
    }
});
