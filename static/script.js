document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const imageInput = document.getElementById('image-input');
    const loader = document.getElementById('loader');
    const resultsDiv = document.getElementById('results');
    const captionOutput = document.getElementById('caption-output');
    const storyOutput = document.getElementById('story-output');

    if (imageInput.files.length === 0) {
        alert('Please select an image file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', imageInput.files[0]);

    loader.style.display = 'block';
    resultsDiv.style.display = 'none';
    captionOutput.textContent = '';
    storyOutput.textContent = '';

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();
        captionOutput.textContent = data.caption;
        storyOutput.textContent = data.story;
        resultsDiv.style.display = 'block';

    } catch (error) {
        alert(`An error occurred: ${error.message}`);
    } finally {
        loader.style.display = 'none';
    }
});