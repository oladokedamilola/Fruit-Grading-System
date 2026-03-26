// Main JavaScript for Fruit Grading System

let currentFile = null;

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    
    // Setup drag and drop
    if (uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        uploadArea.addEventListener('drop', handleDrop, false);
    }
    
    // Handle file input change
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Handle form submission
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleSubmit);
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        if (uploadArea) uploadArea.classList.add('drag-over');
    }
    
    function unhighlight(e) {
        if (uploadArea) uploadArea.classList.remove('drag-over');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }
    
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }
    
    function handleFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (JPG, JPEG, or PNG)');
            return;
        }
        
        // Validate file size (16MB)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            alert('File too large. Maximum size is 16MB');
            return;
        }
        
        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            
            // Hide result if showing
            if (result) result.style.display = 'none';
            
            // Scroll to preview
            previewContainer.scrollIntoView({ behavior: 'smooth' });
        };
        reader.readAsDataURL(file);
        
        // Auto-submit if auto-upload is enabled
        if (window.autoUpload !== false) {
            uploadFile();
        }
    }
    
    async function uploadFile() {
        if (!currentFile) {
            alert('Please select an image first');
            return;
        }
        
        // Show loading, hide result and preview
        loading.style.display = 'block';
        if (result) result.style.display = 'none';
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);
        
        try {
            // Send to server
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide loading
            loading.style.display = 'none';
            
            if (data.success) {
                displayResult(data);
            } else {
                alert('Error: ' + (data.error || 'Unknown error occurred'));
            }
        } catch (error) {
            loading.style.display = 'none';
            console.error('Upload error:', error);
            alert('Error uploading file. Please try again.');
        }
    }
    
    function displayResult(data) {
        // Get result elements
        const resultDiv = document.getElementById('result');
        const fruitTypeSpan = document.getElementById('fruit-type');
        const gradeBadge = document.getElementById('grade-badge');
        const confidenceFill = document.getElementById('confidence-fill');
        const confidencePercent = document.getElementById('confidence-percent');
        const gradeBars = document.getElementById('grade-bars');
        const resultImage = document.getElementById('result-image');
        
        if (!resultDiv) return;
        
        // Update fruit type
        if (fruitTypeSpan) {
            fruitTypeSpan.textContent = data.fruit_type.charAt(0).toUpperCase() + data.fruit_type.slice(1);
        }
        
        // Update grade badge
        if (gradeBadge) {
            gradeBadge.textContent = `Grade ${data.grade}`;
            gradeBadge.className = `grade-badge grade-${data.grade.toLowerCase()}`;
        }
        
        // Update confidence bar
        const confidencePercentValue = (data.confidence * 100).toFixed(1);
        if (confidenceFill) {
            confidenceFill.style.width = `${confidencePercentValue}%`;
        }
        if (confidencePercent) {
            confidencePercent.textContent = `${confidencePercentValue}%`;
        }
        
        // Update grade breakdown
        if (gradeBars && data.grade_confidences) {
            gradeBars.innerHTML = '';
            for (const [grade, score] of Object.entries(data.grade_confidences)) {
                const percent = (score * 100).toFixed(1);
                const gradeItem = document.createElement('div');
                gradeItem.className = 'breakdown-item';
                gradeItem.innerHTML = `
                    <span class="grade-label grade-${grade.toLowerCase()}">Grade ${grade}</span>
                    <div class="breakdown-bar">
                        <div class="breakdown-fill" style="width: ${percent}%"></div>
                    </div>
                    <span class="breakdown-score">${percent}%</span>
                `;
                gradeBars.appendChild(gradeItem);
            }
        }
        
        // Update result image
        if (resultImage && currentFile) {
            const reader = new FileReader();
            reader.onload = function(e) {
                resultImage.src = e.target.result;
            };
            reader.readAsDataURL(currentFile);
        }
        
        // Show result
        resultDiv.style.display = 'block';
        
        // Scroll to result
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Handle form submission
    async function handleSubmit(e) {
        e.preventDefault();
        await uploadFile();
    }
});

// Helper functions
function clearImage() {
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const result = document.getElementById('result');
    
    if (fileInput) fileInput.value = '';
    if (previewContainer) previewContainer.style.display = 'none';
    if (imagePreview) imagePreview.src = '';
    if (result) result.style.display = 'none';
    
    currentFile = null;
}

function resetForm() {
    clearImage();
    const result = document.getElementById('result');
    if (result) result.style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Export functions for global use
window.clearImage = clearImage;
window.resetForm = resetForm;