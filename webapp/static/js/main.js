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
        
        // Check if preview modal exists (for upload.html with modal)
        const previewModal = document.getElementById('preview-modal');
        const modalPreviewImg = document.getElementById('modal-preview-img');
        
        if (previewModal && modalPreviewImg) {
            // Use modal for upload page with preview modal
            const reader = new FileReader();
            reader.onload = function(e) {
                modalPreviewImg.src = e.target.result;
                previewModal.style.display = 'flex';
                previewModal.classList.add('show');
            };
            reader.onerror = function(e) {
                console.error('FileReader error:', e);
                alert('Error reading file. Please try again.');
            };
            reader.readAsDataURL(file);
        } else if (previewContainer && imagePreview) {
            // Fallback to old preview method for other pages
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
                
                // Hide result if showing
                if (result) result.style.display = 'none';
                
                // Scroll to preview
                previewContainer.scrollIntoView({ behavior: 'smooth' });
                
                // Auto-submit if auto-upload is enabled
                if (window.autoUpload !== false) {
                    uploadFile();
                }
            };
            reader.readAsDataURL(file);
        } else {
            // No preview container found, just store the file
            console.log('File selected, but no preview container found');
        }
    }
    
    async function uploadFile() {
        if (!currentFile) {
            alert('Please select an image first');
            return;
        }
        
        // Check if loading element exists
        if (loading) {
            loading.style.display = 'block';
        }
        if (result) {
            result.style.display = 'none';
        }
        
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
            if (loading) {
                loading.style.display = 'none';
            }
            
            if (data.success) {
                displayResult(data);
            } else if (data.limit_reached) {
                // Show login modal if limit reached
                const loginModal = document.getElementById('login-modal');
                if (loginModal) {
                    loginModal.style.display = 'flex';
                } else {
                    alert(data.error || 'Anonymous usage limit reached. Please login or register.');
                }
            } else {
                alert('Error: ' + (data.error || 'Unknown error occurred'));
            }
        } catch (error) {
            if (loading) {
                loading.style.display = 'none';
            }
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
        if (gradeBars && data.confidence_scores) {
            gradeBars.innerHTML = '';
            const grades = ['A', 'B', 'C'];
            for (const grade of grades) {
                const score = data.confidence_scores[grade] || 0;
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
    const previewModal = document.getElementById('preview-modal');
    
    if (fileInput) fileInput.value = '';
    if (previewContainer) previewContainer.style.display = 'none';
    if (imagePreview) imagePreview.src = '';
    if (result) result.style.display = 'none';
    if (previewModal) {
        previewModal.style.display = 'none';
        previewModal.classList.remove('show');
    }
    
    currentFile = null;
}

function resetForm() {
    clearImage();
    const result = document.getElementById('result');
    if (result) result.style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Modal functions for upload page
function closePreviewModal() {
    const modal = document.getElementById('preview-modal');
    if (modal) {
        modal.style.display = 'none';
        modal.classList.remove('show');
    }
    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.value = '';
    currentFile = null;
}

function confirmUpload() {
    closePreviewModal();
    
    // Show loading modal if it exists
    const loadingModal = document.getElementById('loading-modal');
    if (loadingModal) {
        loadingModal.style.display = 'flex';
        
        // Animate loading steps
        updateLoadingStep(0);
        const loadingText = document.getElementById('loading-text');
        if (loadingText) loadingText.textContent = 'Uploading your image...';
        
        setTimeout(() => {
            updateLoadingStep(1);
            if (loadingText) loadingText.textContent = 'AI is analyzing quality...';
        }, 1000);
        
        setTimeout(() => {
            updateLoadingStep(2);
            if (loadingText) loadingText.textContent = 'Generating grade...';
        }, 2000);
    }
    
    uploadFileDirect();
}

function updateLoadingStep(step) {
    const steps = document.querySelectorAll('.loading-steps .step');
    steps.forEach((s, i) => {
        if (i < step) {
            s.classList.add('completed');
            s.classList.remove('active');
        } else if (i === step) {
            s.classList.add('active');
            s.classList.remove('completed');
        } else {
            s.classList.remove('active', 'completed');
        }
    });
}

async function uploadFileDirect() {
    if (!currentFile) return;
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        
        const loadingModal = document.getElementById('loading-modal');
        
        if (data.limit_reached) {
            if (loadingModal) loadingModal.style.display = 'none';
            const loginModal = document.getElementById('login-modal');
            if (loginModal) loginModal.style.display = 'flex';
        } else if (data.error) {
            if (loadingModal) loadingModal.style.display = 'none';
            alert('Error: ' + data.error);
        } else {
            // Store result in sessionStorage to pass to results page
            sessionStorage.setItem('gradingResult', JSON.stringify(data));
            if (currentFile) {
                // Store image as data URL
                const reader = new FileReader();
                reader.onload = function(e) {
                    sessionStorage.setItem('gradingImage', e.target.result);
                    // Redirect to results page
                    window.location.href = '/results';
                };
                reader.readAsDataURL(currentFile);
            } else {
                window.location.href = '/results';
            }
        }
    } catch (error) {
        const loadingModal = document.getElementById('loading-modal');
        if (loadingModal) loadingModal.style.display = 'none';
        alert('Upload failed. Please try again.');
    }
}

function closeLoginModal() {
    const modal = document.getElementById('login-modal');
    if (modal) modal.style.display = 'none';
}

function closeSaveModal() {
    const modal = document.getElementById('save-modal');
    if (modal) modal.style.display = 'none';
}

function showSavePrompt() {
    const modal = document.getElementById('save-modal');
    if (modal) modal.style.display = 'flex';
}

// Export functions for global use
window.clearImage = clearImage;
window.resetForm = resetForm;
window.closePreviewModal = closePreviewModal;
window.confirmUpload = confirmUpload;
window.closeLoginModal = closeLoginModal;
window.closeSaveModal = closeSaveModal;
window.showSavePrompt = showSavePrompt;

// ============================================
// Mobile Navigation Toggle - FIXED VERSION
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    
    if (navToggle && navMenu) {
        // Prevent body scroll when menu is open on mobile
        function preventBodyScroll(shouldPrevent) {
            if (shouldPrevent) {
                document.body.style.overflow = 'hidden';
            } else {
                document.body.style.overflow = '';
            }
        }
        
        // Toggle menu function
        function toggleMenu(show) {
            const isActive = navMenu.classList.contains('active');
            
            if (show === undefined) {
                navMenu.classList.toggle('active');
            } else if (show) {
                navMenu.classList.add('active');
            } else {
                navMenu.classList.remove('active');
            }
            
            const newIsActive = navMenu.classList.contains('active');
            
            // Update icon
            const icon = navToggle.querySelector('i');
            if (newIsActive) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-times');
                navToggle.setAttribute('aria-expanded', 'true');
                preventBodyScroll(true);
            } else {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
                navToggle.setAttribute('aria-expanded', 'false');
                preventBodyScroll(false);
            }
        }
        
        // Toggle button click
        navToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleMenu();
        });
        
        // Close menu when clicking a link
        const navLinks = navMenu.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    toggleMenu(false);
                }
            });
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            if (window.innerWidth <= 768 && navMenu.classList.contains('active')) {
                const isClickInside = navMenu.contains(event.target) || navToggle.contains(event.target);
                if (!isClickInside) {
                    toggleMenu(false);
                }
            }
        });
        
        // Handle window resize
        let resizeTimer;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function() {
                if (window.innerWidth > 768) {
                    if (navMenu.classList.contains('active')) {
                        navMenu.classList.remove('active');
                        const icon = navToggle.querySelector('i');
                        icon.classList.remove('fa-times');
                        icon.classList.add('fa-bars');
                        navToggle.setAttribute('aria-expanded', 'false');
                        preventBodyScroll(false);
                    }
                }
            }, 100);
        });
        
        // Handle escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && navMenu.classList.contains('active')) {
                toggleMenu(false);
            }
        });
    }
});

// ============================================
// Flash Message Auto-Close (fallback)
// ============================================
function autoCloseFlashMessages() {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        setTimeout(() => {
            if (message && message.parentElement) {
                message.classList.add('fade-out');
                setTimeout(() => {
                    if (message && message.parentElement) {
                        message.remove();
                    }
                }, 300);
            }
        }, 7000);
    });
}

// Run on page load
document.addEventListener('DOMContentLoaded', autoCloseFlashMessages);