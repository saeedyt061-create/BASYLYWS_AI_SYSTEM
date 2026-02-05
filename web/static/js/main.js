// SAEED AI System - Main JavaScript

// Utility Functions
const Utils = {
    // Format number with commas
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },
    
    // Format date
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('ar-SA');
    },
    
    // Copy to clipboard
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            console.error('Failed to copy:', err);
            return false;
        }
    },
    
    // Download file
    downloadFile(content, filename, type = 'text/plain') {
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },
    
    // Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Show notification
    showNotification(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alertDiv);
        
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
};

// API Client
const API = {
    baseUrl: '',
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    // Generate code
    generate(data) {
        return this.request('/api/generate', {
            method: 'POST',
            body: data
        });
    },
    
    // Analyze code
    analyze(code) {
        return this.request('/api/analyze', {
            method: 'POST',
            body: { code }
        });
    },
    
    // Fix code
    fix(code, error) {
        return this.request('/api/fix', {
            method: 'POST',
            body: { code, error }
        });
    },
    
    // Optimize code
    optimize(code, type = 'performance') {
        return this.request('/api/optimize', {
            method: 'POST',
            body: { code, type }
        });
    },
    
    // Classify code
    classify(code) {
        return this.request('/api/classify', {
            method: 'POST',
            body: { code }
        });
    },
    
    // Detect vulnerabilities
    detectVulnerabilities(code) {
        return this.request('/api/detect-vulnerabilities', {
            method: 'POST',
            body: { code }
        });
    },
    
    // Get dashboard stats
    getDashboardStats() {
        return this.request('/api/dashboard/stats');
    },
    
    // Get recent activity
    getRecentActivity() {
        return this.request('/api/dashboard/activity');
    },
    
    // Get models info
    getModelsInfo() {
        return this.request('/api/models/info');
    },
    
    // Health check
    health() {
        return this.request('/api/health');
    }
};

// Code Editor Component
class CodeEditor {
    constructor(elementId, options = {}) {
        this.element = document.getElementById(elementId);
        this.options = {
            language: 'python',
            theme: 'light',
            ...options
        };
        
        this.init();
    }
    
    init() {
        if (!this.element) return;
        
        // Add syntax highlighting classes
        this.element.classList.add('code-input');
        
        // Auto-resize
        this.element.addEventListener('input', () => {
            this.autoResize();
        });
    }
    
    autoResize() {
        this.element.style.height = 'auto';
        this.element.style.height = this.element.scrollHeight + 'px';
    }
    
    getValue() {
        return this.element.value;
    }
    
    setValue(value) {
        this.element.value = value;
        this.autoResize();
    }
    
    clear() {
        this.setValue('');
    }
}

// Loading Indicator
class LoadingIndicator {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
    }
    
    show() {
        if (this.element) {
            this.element.style.display = 'block';
        }
    }
    
    hide() {
        if (this.element) {
            this.element.style.display = 'none';
        }
    }
}

// Progress Bar
class ProgressBar {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        this.value = 0;
    }
    
    setValue(value) {
        this.value = Math.min(100, Math.max(0, value));
        if (this.element) {
            this.element.style.width = this.value + '%';
        }
    }
    
    animate(duration = 3000, onComplete = null) {
        const startTime = Date.now();
        
        const update = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(100, (elapsed / duration) * 100);
            
            this.setValue(progress);
            
            if (progress < 100) {
                requestAnimationFrame(update);
            } else if (onComplete) {
                onComplete();
            }
        };
        
        requestAnimationFrame(update);
    }
    
    reset() {
        this.setValue(0);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Health check
    API.health()
        .then(data => {
            console.log('System health:', data);
        })
        .catch(error => {
            console.warn('Health check failed:', error);
        });
});

// Export for use in other scripts
window.SaeedAI = {
    Utils,
    API,
    CodeEditor,
    LoadingIndicator,
    ProgressBar
};
