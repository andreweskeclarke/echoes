// MLflow Architecture Dashboard JavaScript

class ArchitectureDashboard {
    constructor() {
        this.data = null;
        this.selectedModels = new Set();
        this.currentFilter = 'all';
        this.currentSort = 'val_accuracy';
        this.globalMaxParams = 0; // Global maximum for consistent node sizing
        
        this.init();
    }

    async init() {
        try {
            await this.loadData();
            this.setupEventListeners();
            this.renderDashboard();
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showError('Failed to load model data');
        }
    }

    async loadData() {
        const response = await fetch('models_data.json?t=' + Date.now()); // Cache bust
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        this.data = await response.json();
        
        // Calculate global maximum parameters across all models for consistent node sizing
        this.calculateGlobalMaxParams();
    }

    calculateGlobalMaxParams() {
        let maxParams = 0;
        
        for (const experiment of this.data.experiments) {
            // Check input node (112*112*3 for video frames)
            const inputParams = 112 * 112 * 3;
            maxParams = Math.max(maxParams, inputParams);
            
            // Check all architecture layers
            for (const layer of experiment.architecture) {
                maxParams = Math.max(maxParams, layer.params);
            }
            
            // Check output node (number of classes)
            const numClasses = this.getNumClasses(experiment);
            maxParams = Math.max(maxParams, numClasses);
        }
        
        this.globalMaxParams = maxParams;
        console.log('Global max params calculated:', this.globalMaxParams);
    }

    setupEventListeners() {
        const modelTypeFilter = document.getElementById('modelTypeFilter');
        const sortBy = document.getElementById('sortBy');

        modelTypeFilter.addEventListener('change', (e) => {
            this.currentFilter = e.target.value;
            this.renderDashboard();
        });

        sortBy.addEventListener('change', (e) => {
            this.currentSort = e.target.value;
            this.renderDashboard();
        });
    }

    renderDashboard() {
        this.renderStatsSummary();
        this.renderArchitectureGrid();
    }

    renderStatsSummary() {
        const container = document.getElementById('statsSummary');
        const experiments = this.getFilteredExperiments();
        
        const stats = {
            'Total Models': experiments.length,
            'Best Val Accuracy': Math.max(...experiments.map(e => e.metrics.val_accuracy || 0)).toFixed(1) + '%',
            'Avg Training Time': (experiments.reduce((sum, e) => sum + (e.metrics.training_time_seconds || 0), 0) / experiments.length).toFixed(1) + 's',
            'Parameter Range': this.formatParameterRange(experiments)
        };

        container.innerHTML = Object.entries(stats).map(([label, value]) => `
            <div class="stat-card">
                <div class="stat-value">${value}</div>
                <div class="stat-label">${label}</div>
            </div>
        `).join('');
    }

    formatParameterRange(experiments) {
        const trainableParams = experiments.map(e => e.params.trainable_params || 0);
        const min = Math.min(...trainableParams);
        const max = Math.max(...trainableParams);
        return `${this.formatNumber(min)} - ${this.formatNumber(max)}`;
    }

    formatNumber(num) {
        if (num >= 1e12) return (num / 1e12).toFixed(1) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
        return num.toString();
    }

    getFilteredExperiments() {
        let experiments = this.data.experiments;
        
        // Apply filter
        if (this.currentFilter !== 'all') {
            experiments = experiments.filter(e => e.type === this.currentFilter);
        }
        
        // Apply sort
        experiments.sort((a, b) => {
            let aVal, bVal;
            switch (this.currentSort) {
                case 'val_accuracy':
                    aVal = a.metrics.val_accuracy || 0;
                    bVal = b.metrics.val_accuracy || 0;
                    return bVal - aVal; // Descending
                case 'trainable_params':
                    aVal = a.params.trainable_params || 0;
                    bVal = b.params.trainable_params || 0;
                    return aVal - bVal; // Ascending
                case 'total_params':
                    aVal = a.params.total_params || 0;
                    bVal = b.params.total_params || 0;
                    return aVal - bVal; // Ascending
                case 'training_time':
                    aVal = a.metrics.training_time_seconds || 0;
                    bVal = b.metrics.training_time_seconds || 0;
                    return aVal - bVal; // Ascending
                case 'params_per_accuracy':
                    // Lower is better (more efficient)
                    aVal = (a.params.trainable_params || 1) / Math.max((a.metrics.val_accuracy || 0.1), 0.1);
                    bVal = (b.params.trainable_params || 1) / Math.max((b.metrics.val_accuracy || 0.1), 0.1);
                    return aVal - bVal; // Ascending
                case 'model_complexity':
                    // Total layers/components (estimate from architecture length)
                    aVal = a.architecture ? a.architecture.length : 0;
                    bVal = b.architecture ? b.architecture.length : 0;
                    return aVal - bVal; // Ascending
                case 'memory_efficiency':
                    // Accuracy per MB (assuming 4 bytes per parameter)
                    const aMB = ((a.params.total_params || 1) * 4) / (1024 * 1024);
                    const bMB = ((b.params.total_params || 1) * 4) / (1024 * 1024);
                    aVal = (a.metrics.val_accuracy || 0) / aMB;
                    bVal = (b.metrics.val_accuracy || 0) / bMB;
                    return bVal - aVal; // Descending
                case 'training_speed':
                    // Samples per second (train_samples / training_time)
                    aVal = (a.params.train_samples || 1) / Math.max((a.metrics.training_time_seconds || 1), 1);
                    bVal = (b.params.train_samples || 1) / Math.max((b.metrics.training_time_seconds || 1), 1);
                    return bVal - aVal; // Descending
                default:
                    return 0;
            }
        });
        
        return experiments;
    }

    renderArchitectureGrid() {
        const container = document.getElementById('architectureGrid');
        const experiments = this.getFilteredExperiments();
        
        container.innerHTML = experiments.map(experiment => this.createModelCard(experiment)).join('');
        
        // Add click handlers for model selection
        container.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const modelName = e.currentTarget.dataset.modelName;
                this.toggleModelSelection(modelName, e.currentTarget);
            });
        });
    }

    createModelCard(experiment) {
        return `
            <div class="model-card" data-model-name="${experiment.name}">
                <div class="model-header">
                    <div class="model-name">${experiment.name}</div>
                    <div class="model-type">${experiment.type}</div>
                </div>
                
                <div class="model-metrics">
                    <div class="metric">
                        <span class="metric-value">${(experiment.metrics.val_accuracy || 0).toFixed(1)}%</span>
                        Val Accuracy
                    </div>
                    <div class="metric">
                        <span class="metric-value">${(experiment.metrics.training_time_seconds || 0).toFixed(1)}s</span>
                        Training Time
                    </div>
                    <div class="metric">
                        <span class="metric-value">${this.formatNumber(experiment.params.trainable_params || 0)}</span>
                        Trainable Params
                    </div>
                    <div class="metric">
                        <span class="metric-value">${this.formatNumber(experiment.params.total_params || 0)}</span>
                        Total Params
                    </div>
                </div>
                
                <div class="architecture-diagram" id="diagram-${experiment.name}">
                    ${this.createArchitectureDiagram(experiment)}
                </div>
            </div>
        `;
    }

    getNumClasses(experiment) {
        // Try to extract number of classes from the output layer
        const outputLayer = experiment.architecture.find(layer => 
            layer.type === 'output' || layer.name === 'readout' || layer.name === 'fc'
        );
        
        if (outputLayer) {
            // For SimpleRNN: fc_params = hidden_size * num_classes + num_classes
            // For ESN: readout_params = reservoir_size * num_classes + num_classes  
            // For DeepESN: readout_params = (reservoir_size * num_layers) * num_classes + num_classes
            
            if (experiment.type === 'SimpleRNN') {
                const hiddenSize = experiment.params.hidden_size || 64;
                return Math.round(outputLayer.params / (hiddenSize + 1));
            } else if (experiment.type === 'SimpleESN') {
                const reservoirSize = experiment.params.reservoir_size || 500;
                return Math.round(outputLayer.params / (reservoirSize + 1));
            } else if (experiment.type === 'DeepESN') {
                const reservoirSize = experiment.params.reservoir_size || 500;
                const numLayers = experiment.params.num_layers || 2;
                return Math.round(outputLayer.params / (reservoirSize * numLayers + 1));
            }
        }
        
        // Default fallback
        return 25;
    }

    createArchitectureDiagram(experiment) {
        const svgId = `svg-${experiment.name}`;
        const width = 900; // Wider to accommodate input/output nodes
        const height = 300; // Fixed height for horizontal DAG
        
        // Get dynamic number of classes
        const numClasses = this.getNumClasses(experiment);
        
        // Create input and output nodes with actual dimensions
        const inputNode = {
            name: "Input",
            params: 112*112*3, // Image dimensions as parameter count for sizing
            trainable: false,
            type: "input",
            description: "Video frames: 112×112×3",
            displayParams: "112×112×3" // For display purposes
        };
        
        const outputNode = {
            name: "Output", 
            params: numClasses, // Number of classes as parameter count for sizing
            trainable: false,
            type: "output",
            description: `Classes: ${numClasses}`,
            displayParams: numClasses.toString() // For display purposes
        };
        
        // Combine all layers: input + architecture + output
        const allLayers = [inputNode, ...experiment.architecture, outputNode];
        const maxParams = this.globalMaxParams; // Use global maximum for consistent scaling
        
        // Calculate node positions for horizontal DAG layout
        const nodeSpacing = width / (allLayers.length + 1);
        const centerY = height / 2;
        
        const layers = allLayers.map((layer, index) => {
            // Calculate node size based on parameter count (area-based) with linear scaling
            let nodeWidth, nodeHeight;
            
            // Use linear scaling to preserve absolute parameter count differences
            const normalizedSize = Math.max(layer.params, 1) / Math.max(maxParams, 1);
            
            // Pure linear scaling with small minimum to ensure tiny components are visible
            const minArea = 1000;  // Small minimum area for visibility
            const maxArea = 20000; // Maximum area
            const area = minArea + (normalizedSize * (maxArea - minArea));
            
            const aspectRatio = 1.4; // Width/height ratio
            nodeHeight = Math.sqrt(area / aspectRatio);
            nodeWidth = nodeHeight * aspectRatio;
            
            const x = nodeSpacing * (index + 1) - nodeWidth / 2;
            const y = centerY - nodeHeight / 2;
            
            return {
                ...layer,
                width: nodeWidth,
                height: nodeHeight,
                x: x,
                y: y,
                centerX: x + nodeWidth / 2,
                centerY: y + nodeHeight / 2,
                color: this.getLayerColor(layer.type),
                textColor: this.getLayerTextColor(layer.type),
                shape: this.getLayerShape(layer.type)
            };
        });

        // Generate connection paths between layers
        const connections = this.createConnectionPaths(layers);

        const svg = `
            <svg id="${svgId}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
                <defs>
                    ${this.createGradientDefs(layers, experiment.name)}
                    <marker id="arrowhead-${experiment.name}" markerWidth="10" markerHeight="7" 
                            refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#6b7280" />
                    </marker>
                </defs>
                ${connections}
                ${layers.map((layer, index) => this.createDAGNode(layer, index, experiment.name)).join('')}
            </svg>
        `;

        return svg;
    }

    getLayerColor(layerType) {
        const colors = {
            'input': '#3b82f6',      // Blue for input layers
            'reservoir': '#8b5cf6',   // Purple for reservoir layers  
            'connection': '#10b981',  // Green for connection layers
            'rnn': '#ef4444',        // Red for RNN/LSTM layers
            'fc': '#f59e0b',         // Orange for FC layers
            'output': '#f59e0b'      // Orange for output layers
        };
        return colors[layerType] || '#6b7280'; // Default gray
    }

    getLayerTextColor(layerType) {
        return '#ffffff'; // White text for all layer types
    }

    createLayerBlock(layer, index, modelName, containerWidth) {
        const blockWidth = 280;
        const x = (containerWidth - blockWidth) / 2;
        
        // Create gradient for trainable vs frozen indication
        const gradientId = `gradient-${modelName}-${index}`;
        const isTrainable = layer.trainable;
        const baseColor = layer.color;
        const darkColor = this.darkenColor(baseColor, 0.3);
        
        return `
            <defs>
                <linearGradient id="${gradientId}" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:${baseColor};stop-opacity:1" />
                    <stop offset="100%" style="stop-color:${darkColor};stop-opacity:1" />
                </linearGradient>
                ${isTrainable ? `
                <pattern id="trainable-${modelName}-${index}" patternUnits="userSpaceOnUse" width="4" height="4">
                    <rect width="4" height="4" fill="url(#${gradientId})"/>
                    <path d="M 0,4 l 4,-4 M -1,1 l 2,-2 M 3,5 l 2,-2" stroke="${darkColor}" stroke-width="0.5"/>
                </pattern>` : ''}
            </defs>
            
            <g class="layer-block" data-layer="${index}" data-model="${modelName}">
                <!-- Main layer block -->
                <rect x="${x}" y="${layer.y}" width="${blockWidth}" height="${layer.height}" 
                      fill="${isTrainable ? `url(#trainable-${modelName}-${index})` : `url(#${gradientId})`}"
                      stroke="${darkColor}" stroke-width="2" rx="8" 
                      class="layer-rect" />
                
                <!-- Layer type indicator -->
                <rect x="${x}" y="${layer.y}" width="8" height="${layer.height}" 
                      fill="${baseColor}" rx="8 0 0 8" />
                
                <!-- Layer name -->
                <text x="${containerWidth/2}" y="${layer.y + Math.max(20, layer.height * 0.3)}" 
                      fill="${layer.textColor}" text-anchor="middle" 
                      class="layer-name" font-weight="bold" font-size="14">
                    ${layer.name}
                </text>
                
                <!-- Parameter count -->
                <text x="${containerWidth/2}" y="${layer.y + Math.max(35, layer.height * 0.5)}" 
                      fill="#e5e7eb" text-anchor="middle" 
                      class="layer-params" font-size="12">
                    ${this.formatNumber(layer.params)} params
                </text>
                
                <!-- Trainable status -->
                <text x="${containerWidth/2}" y="${layer.y + Math.max(50, layer.height * 0.7)}" 
                      fill="#d1d5db" text-anchor="middle" 
                      class="layer-status" font-size="10">
                    ${isTrainable ? '🔥 TRAINABLE' : '❄️ FROZEN'}
                </text>
                
                <!-- Description (if layer is tall enough) -->
                ${layer.height > 80 ? `
                <text x="${containerWidth/2}" y="${layer.y + layer.height - 15}" 
                      fill="#9ca3af" text-anchor="middle" 
                      class="layer-description" font-size="9">
                    ${layer.description}
                </text>` : ''}
            </g>
        `;
    }

    darkenColor(color, amount) {
        // Convert hex to RGB and darken
        const hex = color.replace('#', '');
        const r = Math.max(0, parseInt(hex.substr(0, 2), 16) - Math.round(255 * amount));
        const g = Math.max(0, parseInt(hex.substr(2, 2), 16) - Math.round(255 * amount));
        const b = Math.max(0, parseInt(hex.substr(4, 2), 16) - Math.round(255 * amount));
        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    }

    toggleModelSelection(modelName, cardElement) {
        if (this.selectedModels.has(modelName)) {
            this.selectedModels.delete(modelName);
            cardElement.classList.remove('selected');
        } else {
            if (this.selectedModels.size < 2) {
                this.selectedModels.add(modelName);
                cardElement.classList.add('selected');
            }
        }
        
        this.updateComparisonView();
    }

    updateComparisonView() {
        const comparisonView = document.getElementById('comparisonView');
        
        if (this.selectedModels.size === 2) {
            comparisonView.style.display = 'block';
            const models = Array.from(this.selectedModels).map(name => 
                this.data.experiments.find(e => e.name === name)
            );
            
            document.getElementById('modelA').innerHTML = this.createComparisonModel(models[0]);
            document.getElementById('modelB').innerHTML = this.createComparisonModel(models[1]);
        } else {
            comparisonView.style.display = 'none';
        }
    }

    createComparisonModel(experiment) {
        return `
            <h3>${experiment.name}</h3>
            <div class="architecture-diagram">
                ${this.createArchitectureDiagram(experiment)}
            </div>
            <div class="comparison-metrics">
                <p><strong>Val Accuracy:</strong> ${(experiment.metrics.val_accuracy || 0).toFixed(1)}%</p>
                <p><strong>Trainable Params:</strong> ${this.formatNumber(experiment.params.trainable_params || 0)}</p>
                <p><strong>Training Time:</strong> ${(experiment.metrics.training_time_seconds || 0).toFixed(1)}s</p>
                <p><strong>Efficiency:</strong> ${((experiment.params.trainable_params || 0) / Math.max(experiment.metrics.val_accuracy || 0.1, 0.1)).toFixed(0)} params/acc</p>
            </div>
        `;
    }

    getLayerShape(layerType) {
        // Always use rounded rectangles for consistency
        return 'rect';
    }

    createConnectionPaths(layers) {
        let paths = '';
        for (let i = 0; i < layers.length - 1; i++) {
            const current = layers[i];
            const next = layers[i + 1];
            
            const startX = current.x + current.width;
            const startY = current.centerY;
            const endX = next.x;
            const endY = next.centerY;
            
            // Create curved connection path
            const midX = (startX + endX) / 2;
            const path = `M ${startX} ${startY} Q ${midX} ${startY} ${midX} ${(startY + endY) / 2} Q ${midX} ${endY} ${endX} ${endY}`;
            
            paths += `
                <path d="${path}" stroke="#6b7280" stroke-width="2" fill="none" 
                      opacity="0.6" class="connection-path" 
                      marker-end="url(#arrowhead-${layers[0].name || 'default'})" />
            `;
        }
        return paths;
    }

    createGradientDefs(layers, modelName) {
        let defs = '';
        layers.forEach((layer, index) => {
            const gradientId = `gradient-${modelName}-${index}`;
            const baseColor = layer.color;
            const darkColor = this.darkenColor(baseColor, 0.3);
            
            defs += `
                <linearGradient id="${gradientId}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:${baseColor};stop-opacity:1" />
                    <stop offset="100%" style="stop-color:${darkColor};stop-opacity:1" />
                </linearGradient>
            `;
            
            if (layer.trainable) {
                defs += `
                    <pattern id="trainable-${modelName}-${index}" patternUnits="userSpaceOnUse" width="4" height="4">
                        <rect width="4" height="4" fill="url(#${gradientId})"/>
                        <path d="M 0,4 l 4,-4 M -1,1 l 2,-2 M 3,5 l 2,-2" stroke="${darkColor}" stroke-width="0.5"/>
                    </pattern>
                `;
            }
        });
        return defs;
    }

    createDAGNode(layer, index, modelName) {
        const gradientId = `gradient-${modelName}-${index}`;
        const shape = layer.shape;
        const fillStyle = layer.trainable ? `url(#trainable-${modelName}-${index})` : `url(#${gradientId})`;
        
        // Always use rounded rectangles for consistency
        const nodeElement = `
            <rect x="${layer.x}" y="${layer.y}" width="${layer.width}" height="${layer.height}" 
                  fill="${fillStyle}" stroke="${this.darkenColor(layer.color, 0.3)}" 
                  stroke-width="2" rx="12" class="dag-node" />
        `;
        
        // Add text labels
        const textX = layer.centerX;
        const textY = layer.centerY;
        
        const textElements = `
            <text x="${textX}" y="${textY - 10}" fill="${layer.textColor}" text-anchor="middle" 
                  class="dag-node-name" font-weight="bold" font-size="14">
                ${layer.name.length > 10 ? layer.name.substring(0, 10) + '...' : layer.name}
            </text>
            <text x="${textX}" y="${textY + 8}" fill="#e5e7eb" text-anchor="middle" 
                  class="dag-node-params" font-size="12">
                ${layer.displayParams || this.formatNumber(layer.params)}
            </text>
            <text x="${textX}" y="${textY + 24}" fill="#d1d5db" text-anchor="middle" 
                  class="dag-node-status" font-size="10">
                ${layer.trainable ? '🔥' : '❄️'}
            </text>
        `;
        
        return `
            <g class="dag-node-group" data-layer="${index}" data-model="${modelName}">
                ${nodeElement}
                ${textElements}
            </g>
        `;
    }

    showError(message) {
        const container = document.getElementById('architectureGrid');
        container.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: #ff6b6b;">
                <h3>Error Loading Dashboard</h3>
                <p>${message}</p>
                <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.8;">
                    Make sure models_data.json is available and properly formatted.
                </p>
            </div>
        `;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ArchitectureDashboard();
});

// Add tooltip functionality
document.addEventListener('mouseover', (e) => {
    if (e.target.closest('.layer')) {
        const tooltip = document.getElementById('tooltip');
        const layer = e.target.closest('.layer');
        const layerIndex = layer.dataset.layer;
        const modelName = layer.dataset.model;
        
        // Find the experiment and layer data
        // This would need to be implemented based on your data structure
        tooltip.innerHTML = `
            <strong>Layer Details</strong><br>
            Model: ${modelName}<br>
            Layer: ${layerIndex}<br>
            Click for more details
        `;
        
        tooltip.style.left = e.pageX + 10 + 'px';
        tooltip.style.top = e.pageY + 10 + 'px';
        tooltip.classList.add('visible');
    }
});

document.addEventListener('mouseout', (e) => {
    if (!e.relatedTarget || !e.relatedTarget.closest('.layer')) {
        document.getElementById('tooltip').classList.remove('visible');
    }
});
