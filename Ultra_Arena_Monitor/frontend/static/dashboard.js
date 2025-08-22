const { useState, useEffect, useRef } = React;

function Dashboard() {
    const [chartData, setChartData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [files, setFiles] = useState([]);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [layoutConfig, setLayoutConfig] = useState({ charts_per_row: 4 });
    const [monitoringStatus, setMonitoringStatus] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(false); // Start with auto-refresh OFF
    const [refreshInterval, setRefreshInterval] = useState(null);
    const [tooltip, setTooltip] = useState({ show: false, content: '', x: 0, y: 0 });
    
    // Track Latest state
    const [trackLatest, setTrackLatest] = useState(true); // Default to ON as requested
    const [currentDirectory, setCurrentDirectory] = useState(null);
    
    // Animation and scaling state
    const [chartScales, setChartScales] = useState({});
    const [animationData, setAnimationData] = useState({});
    const animationRefs = useRef({});
    const previousData = useRef({});

    useEffect(() => {
        fetchInitialData();
        fetchMonitoringStatus();
        
        // Don't set up auto-refresh initially - let user toggle it on

        // Add window resize listener for responsive chart dimensions
        const handleResize = () => {
            // Force re-render of charts when window is resized
            setChartData(prevData => [...prevData]);
        };

        window.addEventListener('resize', handleResize);

        return () => {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
            window.removeEventListener('resize', handleResize);
        };
    }, []); // Remove autoRefresh dependency to prevent re-renders

    useEffect(() => {
        // Update CSS custom property for charts per row
        const chartsGrid = document.querySelector('.charts-grid');
        if (chartsGrid && layoutConfig.charts_per_row) {
            chartsGrid.style.setProperty('--charts-per-row', layoutConfig.charts_per_row);
        }
    }, [layoutConfig]);

    const fetchInitialData = async () => {
        try {
            setLoading(true);
            const [chartResponse, filesResponse, layoutResponse] = await Promise.all([
                fetch('/api/chart-data'),
                fetch('/api/files'),
                fetch('/api/layout-config')
            ]);

            if (!chartResponse.ok || !filesResponse.ok || !layoutResponse.ok) {
                throw new Error('Failed to fetch data');
            }

            const chartDataResult = await chartResponse.json();
            const filesResult = await filesResponse.json();
            const layoutResult = await layoutResponse.json();

            // Handle case where no data is available yet
            if (!chartDataResult || chartDataResult.length === 0) {
                setLoading(false);
                setError('No chart data available yet. Waiting for JSON files to be processed...');
                return;
            }

            // Initialize scales and animation data
            const initialScales = {};
            const initialAnimationData = {};
            
            console.log('Initial chart data:', chartDataResult);
            
            chartDataResult.forEach(chart => {
                const chartId = chart.chart_title;
                const maxValue = Math.max(...chart.datasets.map(d => d.value || 0));
                
                console.log(`Setting up chart: ${chartId}, maxValue: ${maxValue}`);
                
                // Set initial scale based on chart type
                if (chart.chart_title.includes('Percent') || chart.chart_title.includes('Percentage')) {
                    initialScales[chartId] = 100; // Percentage charts: 0-100
                } else {
                    // For other charts, start with 2x the max value or a reasonable default
                    initialScales[chartId] = Math.max(maxValue * 2, 1000);
                }
                
                console.log(`Scale for ${chartId}: ${initialScales[chartId]}`);
                
                // Initialize animation data with 0 values
                initialAnimationData[chartId] = chart.datasets.map(d => ({
                    ...d,
                    animatedValue: 0,
                    targetValue: d.value || 0
                }));
                
                console.log(`Animation data for ${chartId}:`, initialAnimationData[chartId]);
            });

            setChartScales(initialScales);
            setAnimationData(initialAnimationData);
            previousData.current = chartDataResult;
            
            // Sort and set data
            const sortedFiles = sortFiles(filesResult);
            const sortedChartData = sortChartData(chartDataResult);
            
            setChartData(sortedChartData);
            setFiles(sortedFiles);
            setLayoutConfig(layoutResult);
            setError(null);
            setLoading(false);
            
            // Trigger initial animation after a short delay to ensure state is set
            setTimeout(() => {
                console.log('Triggering initial animation...');
                animateCharts();
            }, 100);
            
        } catch (err) {
            console.error('Error fetching initial data:', err);
            setError('Failed to load dashboard data. Please check if the server is running and try again.');
            setLoading(false);
        }
    };

    const fetchChartDataOnly = async () => {
        try {
            const response = await fetch('/api/chart-data');
            if (!response.ok) {
                console.warn('Failed to fetch chart data, keeping existing data');
                return;
            }

            const newChartData = await response.json();
            
            // If no new data is available, keep existing data
            if (!newChartData || newChartData.length === 0) {
                console.log('No new chart data available, keeping existing data');
                return;
            }
            
            const sortedChartData = sortChartData(newChartData);
            
            // Update animation data with new target values
            const updatedAnimationData = { ...animationData };
            const updatedScales = { ...chartScales };
            
            sortedChartData.forEach(chart => {
                const chartId = chart.chart_title;
                const currentAnimationData = updatedAnimationData[chartId] || [];
                
                // Update target values and check for scale adjustments
                const newMaxValue = Math.max(...chart.datasets.map(d => d.value || 0));
                const currentScale = updatedScales[chartId];
                
                // Check if we need to increase scale (if tallest bar reaches 75% of current scale)
                if (newMaxValue > currentScale * 0.75 && !chart.chart_title.includes('Percent')) {
                    updatedScales[chartId] = currentScale * 2;
                }
                
                // Update animation targets
                updatedAnimationData[chartId] = chart.datasets.map((d, index) => ({
                    ...d,
                    animatedValue: currentAnimationData[index]?.animatedValue || 0,
                    targetValue: d.value || 0
                }));
            });
            
            setChartScales(updatedScales);
            setAnimationData(updatedAnimationData);
            setChartData(sortedChartData);
            setLastUpdate(new Date().toLocaleTimeString());
            
            // Animate to new values
            animateCharts();
            
        } catch (err) {
            console.error('Error fetching chart data:', err);
            // Don't update state on error, keep existing data
        }
    };

    const fetchFilesOnly = async () => {
        try {
            const response = await fetch('/api/files');
            if (!response.ok) {
                console.warn('Failed to fetch files data, keeping existing data');
                return;
            }
            const newFiles = await response.json();
            const sortedFiles = sortFiles(newFiles);
            
            // Only update if files have actually changed
            const currentFilesString = JSON.stringify(files);
            const newFilesString = JSON.stringify(sortedFiles);
            
            if (currentFilesString !== newFilesString) {
                setFiles(sortedFiles);
                setLastUpdate(new Date().toLocaleTimeString()); // Update timestamp
                console.log('Files updated:', sortedFiles.length, 'files');
            } else {
                console.log('No changes in files data');
            }
        } catch (err) {
            console.error('Error fetching files data:', err);
        }
    };

    const animateCharts = () => {
        console.log('Starting chart animations...');
        console.log('Current animation data:', animationData);
        
        Object.keys(animationData).forEach(chartId => {
            const chartAnimationData = animationData[chartId];
            if (!chartAnimationData) return;
            
            console.log(`Animating chart: ${chartId}`, chartAnimationData);
            
            chartAnimationData.forEach((dataset, index) => {
                const currentValue = dataset.animatedValue;
                const targetValue = dataset.targetValue;
                
                console.log(`Dataset ${index}: current=${currentValue}, target=${targetValue}`);
                
                // Start animation if values are different (not just if difference > 0.1)
                if (Math.abs(currentValue - targetValue) > 0.01) {
                    console.log(`Starting animation for ${chartId} dataset ${index}: ${currentValue} -> ${targetValue}`);
                    animateValue(chartId, index, currentValue, targetValue);
                } else {
                    console.log(`No animation needed for ${chartId} dataset ${index}: values are the same`);
                }
            });
        });
    };

    const animateValue = (chartId, datasetIndex, startValue, endValue) => {
        console.log(`animateValue called: ${chartId}[${datasetIndex}] ${startValue} -> ${endValue}`);
        
        const duration = 1000; // 1 second animation
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentValue = startValue + (endValue - startValue) * easeOutQuart;
            
            console.log(`Animation progress: ${progress.toFixed(2)}, currentValue: ${currentValue.toFixed(2)}`);
            
            // Update animation data
            setAnimationData(prev => ({
                ...prev,
                [chartId]: prev[chartId].map((dataset, index) => 
                    index === datasetIndex 
                        ? { ...dataset, animatedValue: currentValue }
                        : dataset
                )
            }));
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                console.log(`Animation completed for ${chartId}[${datasetIndex}]`);
            }
        };
        
        requestAnimationFrame(animate);
    };

    const sortFiles = (files) => {
        return files.sort((a, b) => {
            const providerComparison = a.llm_provider.localeCompare(b.llm_provider);
            if (providerComparison !== 0) {
                return providerComparison;
            }
            return a.llm_model.localeCompare(b.llm_model);
        });
    };

    const sortChartData = (chartData) => {
        return chartData.map(chart => ({
            ...chart,
            datasets: chart.datasets.sort((a, b) => {
                const aParts = a.label.split('_');
                const bParts = b.label.split('_');
                
                if (aParts.length >= 2 && bParts.length >= 2) {
                    const aProvider = aParts[0];
                    const bProvider = bParts[0];
                    const aModel = aParts[1];
                    const bModel = bParts[1];
                    
                    const providerComparison = aProvider.localeCompare(bProvider);
                    if (providerComparison !== 0) {
                        return providerComparison;
                    }
                    return aModel.localeCompare(bModel);
                }
                
                return a.label.localeCompare(b.label);
            })
        }));
    };

    const fetchMonitoringStatus = async () => {
        try {
            const response = await fetch('/api/monitoring-status');
            if (response.ok) {
                const status = await response.json();
                setMonitoringStatus(status);
                
                // Update track latest state from server
                if (status.track_latest_enabled !== undefined) {
                    setTrackLatest(status.track_latest_enabled);
                }
                if (status.current_json_data_dir) {
                    setCurrentDirectory(status.current_json_data_dir);
                }
            }
        } catch (err) {
            console.error('Failed to fetch monitoring status:', err);
        }
    };

    const toggleTrackLatest = async () => {
        try {
            const newState = !trackLatest;
            const response = await fetch('/api/track-latest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: newState })
            });
            
            if (response.ok) {
                const result = await response.json();
                setTrackLatest(newState);
                if (result.current_directory) {
                    setCurrentDirectory(result.current_directory);
                }
                
                // Refresh data with new directory
                await fetchInitialData();
                await fetchFilesOnly();
                await fetchMonitoringStatus();
                
                console.log('Track latest toggled:', newState ? 'ON' : 'OFF');
            } else {
                console.error('Failed to toggle track latest');
            }
        } catch (error) {
            console.error('Failed to toggle track latest:', error);
        }
    };

    const toggleAutoRefresh = (event) => {
        // Prevent any default form submission behavior
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        
        const newAutoRefreshState = !autoRefresh;
        setAutoRefresh(newAutoRefreshState);
        
        // Clear existing interval
        if (refreshInterval) {
            clearInterval(refreshInterval);
            setRefreshInterval(null);
        }
        
        // If turning on auto-refresh, start the interval
        if (newAutoRefreshState) {
            const interval = setInterval(() => {
                fetchChartDataOnly();
                fetchFilesOnly();
                fetchMonitoringStatus();
            }, 2000); // Changed from 1000ms to 2000ms to reduce flashing
            setRefreshInterval(interval);
        }
        
        console.log('Auto-refresh toggled:', newAutoRefreshState ? 'ON' : 'OFF');
    };

    const forceRefresh = async () => {
        try {
            await fetch('/api/force-refresh');
            await fetchInitialData();
            await fetchFilesOnly(); // Also refresh files
            setLastUpdate(new Date().toLocaleTimeString());
        } catch (err) {
            console.error('Failed to force refresh:', err);
        }
    };

    const generateDistinctColors = (count) => {
        const colors = [];
        
        // Moderately soft, sophisticated color palette
        const professionalColors = [
            '#4A5F7A', // Medium Blue-Gray
            '#8E44AD', // Medium Purple
            '#E67E22', // Medium Orange
            '#27AE60', // Medium Green
            '#C0392B', // Medium Red
            '#2980B9', // Medium Blue
            '#16A085', // Medium Teal
            '#F39C12', // Medium Yellow-Orange
            '#7F8C8D', // Medium Gray
            '#A0522D', // Medium Brown
            '#2E8B57', // Medium Mint
            '#FF6B6B', // Soft Red
            '#4ECDC4', // Soft Teal
            '#45B7D1', // Soft Blue
            '#96CEB4', // Soft Green
            '#FFEAA7', // Soft Yellow
            '#DDA0DD', // Soft Purple
            '#98D8C8', // Soft Mint
            '#F7DC6F', // Soft Gold
            '#BB8FCE'  // Soft Lavender
        ];
        
        for (let i = 0; i < count; i++) {
            colors.push(professionalColors[i % professionalColors.length]);
        }
        
        return colors;
    };

    const getColorForIndex = (index, totalCount) => {
        const colors = generateDistinctColors(totalCount);
        return colors[index % colors.length];
    };

    // Helper function to format values based on chart configuration
    const formatValue = (value, fieldName, chartInfo) => {
        // Get decimal places from chart configuration
        const decimalPlaces = chartInfo?.decimal_places;
        
        if (decimalPlaces === 0) {
            return Math.round(value).toString();
        } else if (decimalPlaces > 0) {
            return value.toFixed(decimalPlaces);
        } else {
            // If no DECIMAL attribute is specified, show as integer (no decimal places)
            return Math.round(value).toString();
        }
    };

    // Helper function to create short, readable strategy names
    const getShortStrategyName = (fullStrategyName) => {
        // Remove common prefixes and make it more readable
        let shortName = fullStrategyName
            .replace(/^grp_/, '') // Remove grp_ prefix
            .replace(/_/g, ' ') // Replace underscores with spaces
            .replace(/\b\w/g, l => l.toUpperCase()); // Capitalize first letter of each word
        
        // Further shorten common patterns
        shortName = shortName
            .replace(/Openai Gpt4/g, 'GPT-4')
            .replace(/Openai Gpt3/g, 'GPT-3')
            .replace(/Claude Sonnet 4/g, 'Claude Sonnet')
            .replace(/Google Gemini/g, 'Gemini')
            .replace(/Text First/g, 'Text')
            .replace(/Image First/g, 'Image')
            .replace(/Direct File/g, 'Direct')
            .replace(/Parallel/g, 'Para');
        
        return shortName;
    };

    // Function to find file data for a strategy
    const findFileDataForStrategy = (strategyName) => {
        return files.find(file => file.strategy === strategyName);
    };

    // Function to find file data using file_name for perfect matching
    const findFileDataByFileName = (fileName) => {
        return files.find(file => file.file_name === fileName);
    };

    // Enhanced function to find file data using multiple matching criteria
    const findFileDataForChartDataset = (dataset) => {
        // First try to match by strategy (most reliable)
        let fileData = files.find(file => file.strategy === dataset.strategy);
        
        // If no match by strategy, try to match by the detailed label
        if (!fileData) {
            fileData = files.find(file => {
                const fileLabel = `${file.llm_provider} - ${file.llm_model} - ${file.strategy} - ${file.mode}`;
                return fileLabel === dataset.label;
            });
        }
        
        // If still no match, try to match by provider and model
        if (!fileData) {
            fileData = files.find(file => 
                file.llm_provider === dataset.llm_provider && 
                file.llm_model === dataset.llm_model
            );
        }
        
        return fileData;
    };

    // Helper function to calculate responsive chart dimensions
    const getChartDimensions = () => {
        const screenWidth = window.innerWidth;
        
        if (screenWidth <= 480) {
            return { width: 396.75, height: 297.56 }; // 4:3 ratio for mobile - another 15% larger
        } else if (screenWidth <= 768) {
            return { width: 495.94, height: 371.96 }; // 4:3 ratio for tablet - another 15% larger
        } else if (screenWidth <= 1200) {
            return { width: 595.13, height: 446.35 }; // 4:3 ratio for small desktop - another 15% larger
        } else {
            return { width: 694.31, height: 520.73 }; // 4:3 ratio for large desktop - another 15% larger
        }
    };

    const renderAnimatedBarChart = (chartInfo) => {
        const chartId = chartInfo.chart_title;
        const animationDataForChart = animationData[chartId] || chartInfo.datasets;
        
        // Sort datasets to match the order of files in the Available Data Files section
        const sortedDatasets = [...animationDataForChart].sort((a, b) => {
            const aIndex = files.findIndex(f => f.file_name === a.file_name);
            const bIndex = files.findIndex(f => f.file_name === b.file_name);
            return aIndex - bIndex; // Sort by file index to match files array order
        });
        
        // Use consistent dimensions for all charts
        const chartWidth = 600;
        const chartHeight = 400;
        const margin = { top: 5, right: 5, bottom: 10, left: 5 }; // Much smaller margins
        const barPadding = 16;
        const BAR_THRESHOLD = 6; // Configurable threshold for 45-degree rotation
        
        const colors = generateDistinctColors(sortedDatasets.length);
        
        // Calculate the actual maximum value from this chart's data
        const maxValue = Math.max(...sortedDatasets.map(d => d.value || 0));
        const scaleValue = maxValue > 0 ? maxValue * 1.1 : 100; // Add 10% padding for visual appeal
        
        return (
            <div key={chartId} className="chart-container">
                <div className="chart-title">{chartInfo.chart_title}</div>
                <div className="chart-wrapper">
                    <svg width="100%" height="100%" style={{ overflow: 'visible' }} viewBox={`0 0 ${chartWidth} ${chartHeight}`} preserveAspectRatio="xMidYMid meet">
                        {sortedDatasets.map((dataset, index) => {
                            const value = dataset.value || 0;
                            
                            // Find the corresponding file data for color consistency
                            const fileData = findFileDataByFileName(dataset.file_name);
                            const fileIndex = fileData ? files.findIndex(f => f.file_name === dataset.file_name) : index;
                            
                            // Calculate initial bar heights for all datasets in this chart
                            const maxValue = Math.max(...sortedDatasets.map(d => d.value || 0));
                            const availableHeight = chartHeight - margin.top - margin.bottom - 80; // Reserve space for value label
                            
                            // Calculate scale factor if any bar would exceed MAX_BAR_HEIGHT
                            const maxBarHeightConfig = layoutConfig?.max_bar_height || 300;
                            const maxCalculatedHeight = maxValue > 0 ? (maxValue / scaleValue) * availableHeight : 0;
                            const scaleFactor = maxCalculatedHeight > maxBarHeightConfig ? maxBarHeightConfig / maxCalculatedHeight : 1;
                            
                            // Use actual data scale for correct relative bar heights with scaling
                            let barHeight = (value / scaleValue) * availableHeight * scaleFactor;
                            
                            // Ensure minimum bar height for visibility
                            if (barHeight < (layoutConfig?.min_bar_height || 3)) {
                                barHeight = layoutConfig?.min_bar_height || 3;
                            }
                            
                            const barX = margin.left + index * ((chartWidth - margin.left - margin.right - (sortedDatasets.length - 1) * barPadding) / sortedDatasets.length + barPadding);
                            const barY = margin.top + (chartHeight - margin.top - margin.bottom) - barHeight;
                            const barWidth = (chartWidth - margin.left - margin.right - (sortedDatasets.length - 1) * barPadding) / sortedDatasets.length;
                            
                            // Check if value is non-integer and if we have more than threshold bars
                            const isNonInteger = !Number.isInteger(value);
                            const shouldRotate = sortedDatasets.length > BAR_THRESHOLD; // Apply to all labels when > 6 bars
                            
                            // Create enhanced tooltip content with HTML for colored value
                            const shortStrategyName = getShortStrategyName(dataset.label);
                            const formattedValue = formatValue(value, chartInfo.field_name, chartInfo);
                            const tooltipContent = fileData 
                                ? `- ${fileData.llm_provider}\n- ${fileData.llm_model}\n- ${fileData.strategy}\n- ${fileData.mode}\n- <span style="color: #00ff00; font-weight: bold;">${formattedValue}</span>`
                                : `- ${dataset.label}\n- <span style="color: #00ff00; font-weight: bold;">${formattedValue}</span>`;
                            
                            return (
                                <g key={index}>
                                    {/* Bar with improved styling - larger and more visible */}
                                    <rect
                                        x={barX}
                                        y={barY}
                                        width={barWidth}
                                        height={barHeight}
                                        fill={colors[fileIndex % colors.length]}
                                        rx="6"
                                        ry="6"
                                        style={{ 
                                            cursor: 'pointer',
                                            transition: 'all 0.3s ease',
                                            filter: 'drop-shadow(0 3px 6px rgba(0,0,0,0.15))'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.target.style.opacity = '0.8';
                                            e.target.style.filter = 'drop-shadow(0 6px 12px rgba(0,0,0,0.25))';
                                            // Show tooltip
                                            setTooltip({
                                                show: true,
                                                content: tooltipContent,
                                                x: e.clientX + 10,
                                                y: e.clientY - 10
                                            });
                                        }}
                                        onMouseLeave={(e) => {
                                            e.target.style.opacity = '1';
                                            e.target.style.filter = 'drop-shadow(0 3px 6px rgba(0,0,0,0.15))';
                                            // Hide tooltip
                                            setTooltip({ show: false, content: '', x: 0, y: 0 });
                                        }}
                                        onMouseMove={(e) => {
                                            // Update tooltip position as mouse moves
                                            setTooltip(prev => ({
                                                ...prev,
                                                x: e.clientX + 10,
                                                y: e.clientY - 10
                                            }));
                                        }}
                                    />
                                    
                                    {/* Value label with improved styling and conditional rotation */}
                                    <text
                                        x={barX + barWidth / 2 + (layoutConfig?.val_legend_right_bias || 10)}
                                        y={barY - 20}
                                        textAnchor="middle"
                                        fontSize="24"
                                        fill="#2c3e50"
                                        fontWeight="600"
                                        style={{
                                            textShadow: '0 1px 2px rgba(255,255,255,0.8)',
                                            pointerEvents: 'none',
                                            transform: shouldRotate ? 'rotate(-45deg)' : 'none',
                                            transformOrigin: `${barX + barWidth / 2 + (layoutConfig?.val_legend_right_bias || 10)}px ${barY - 20}px`
                                        }}
                                    >
                                        {formatValue(value, chartInfo.field_name, chartInfo)}
                                    </text>
                                </g>
                            );
                        })}
                    </svg>
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="container">
                <div className="loading">
                    <h2>Loading dashboard data...</h2>
                    <p>Please wait while we fetch the comparison data.</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container">
                <div className="error">
                    <h3>Error loading data</h3>
                    <p>{error}</p>
                    <button onClick={fetchInitialData} style={{
                        marginTop: '10px',
                        padding: '10px 20px',
                        backgroundColor: '#667eea',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: 'pointer'
                    }}>
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="container">
            {/* Custom Tooltip */}
            {tooltip.show && (
                <div style={{
                    position: 'fixed',
                    left: tooltip.x,
                    top: tooltip.y,
                    backgroundColor: 'rgba(0, 0, 0, 0.6)', // Increased transparency by 25%
                    color: 'white',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '10.5px', // Reduced by 25% from 14px
                    fontWeight: '500',
                    zIndex: 1000,
                    pointerEvents: 'none',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                    border: '1px solid rgba(255,255,255,0.2)',
                    whiteSpace: 'pre-line',
                    lineHeight: '1.3'
                }}
                dangerouslySetInnerHTML={{ __html: tooltip.content }}
                />
            )}
            
            {/* Header Panel */}
            <div className="header">
                <h1>LLM Performance Comparison Dashboard</h1>
                <p>Comparing results across different LLM providers and models</p>
                
                {/* Real-time Monitoring Status */}
                <div className="monitoring-controls" style={{
                    marginTop: '20px',
                    padding: '15px'
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                            <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)' }}>
                                📁 {monitoringStatus?.json_files_count || 0} files | 
                                ⏱️ {monitoringStatus?.update_frequency_seconds || 1}s updates |
                                🕐 Last: {lastUpdate || 'Never'}
                            </div>
                        </div>
                        
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button 
                                type="button"
                                onClick={toggleTrackLatest}
                                style={{
                                    padding: '8px 16px',
                                    backgroundColor: trackLatest ? '#17a2b8' : '#6c757d',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer',
                                    fontSize: '14px'
                                }}
                            >
                                {trackLatest ? '🎯 Track Latest ON' : '📁 Track Latest OFF'}
                            </button>
                            
                            <button 
                                type="button"
                                onClick={(e) => toggleAutoRefresh(e)}
                                style={{
                                    padding: '8px 16px',
                                    backgroundColor: autoRefresh ? '#28a745' : '#6c757d',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer',
                                    fontSize: '14px'
                                }}
                            >
                                {autoRefresh ? '🔄 Auto-Refresh ON' : '⏸️ Auto-Refresh OFF'}
                            </button>
                            
                            <button 
                                type="button"
                                onClick={forceRefresh}
                                style={{
                                    padding: '8px 16px',
                                    backgroundColor: '#007bff',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer',
                                    fontSize: '14px'
                                }}
                            >
                                🔄 Force Refresh
                            </button>
                        </div>
                    </div>
                    
                    {/* Status Message */}
                    {monitoringStatus && monitoringStatus.status_message && (
                        <div style={{ 
                            marginTop: '10px', 
                            padding: '8px 12px',
                            borderRadius: '5px',
                            fontSize: '14px',
                            fontWeight: '500',
                            backgroundColor: monitoringStatus.status_type === 'warning' ? 'rgba(255, 193, 7, 0.2)' : 
                                           monitoringStatus.status_type === 'info' ? 'rgba(13, 202, 240, 0.2)' : 
                                           'rgba(40, 167, 69, 0.2)',
                            border: `1px solid ${
                                monitoringStatus.status_type === 'warning' ? '#ffc107' : 
                                monitoringStatus.status_type === 'info' ? '#0dcaf0' : 
                                '#28a745'
                            }`,
                            color: monitoringStatus.status_type === 'warning' ? '#ffc107' : 
                                   monitoringStatus.status_type === 'info' ? '#0dcaf0' : 
                                   '#28a745'
                        }}>
                            {monitoringStatus.status_message}
                        </div>
                    )}
                    
                    {monitoringStatus && (
                        <>
                            {/* Beautiful white line partition */}
                            <div style={{
                                marginTop: '15px',
                                marginBottom: '10px',
                                height: '1px',
                                background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 20%, rgba(255,255,255,0.6) 50%, rgba(255,255,255,0.3) 80%, transparent 100%)',
                                borderRadius: '1px'
                            }}></div>
                            
                            <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.7)' }}>
                                📂 Directory: {currentDirectory || monitoringStatus.json_directory} | 
                                {trackLatest ? ' 🎯 Tracking Latest' : ' 📁 Tracking Specific'} |
                                {monitoringStatus.files_changed ? ' 🔄 Files changed' : ' ✅ No changes'} |
                                {monitoringStatus.has_cached_data ? ' 💾 Using cached data' : ' 📭 No cached data'}
                            </div>
                        </>
                    )}
                </div>
            </div>

            {files.length > 0 && (
                <div className="stats-summary">
                    <div className="stats-title">
                        Available Data Files
                        {lastUpdate && (
                            <span style={{
                                fontSize: '0.8rem',
                                fontWeight: 'normal',
                                color: '#666',
                                marginLeft: '10px'
                            }}>
                                (Last updated: {lastUpdate})
                            </span>
                        )}
                    </div>
                    <div className="stats-grid">
                        {files.map((file, index) => {
                            return (
                                <div 
                                    key={index} 
                                    className="stat-item"
                                    style={{
                                        backgroundColor: getColorForIndex(index, files.length),
                                        color: 'white',
                                        borderLeft: `2px solid ${getColorForIndex(index, files.length)}`,
                                        padding: '7px'
                                    }}
                                >
                                    <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>LLM Provider</div>
                                    <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.llm_provider}</div>
                                    <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>Model</div>
                                    <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.llm_model}</div>
                                    <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>Strategy</div>
                                    <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.strategy}</div>
                                    <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>Mode</div>
                                    <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.mode}</div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            <div className="charts-grid">
                {chartData.map((chartInfo, index) => renderAnimatedBarChart(chartInfo))}
            </div>
        </div>
    );
}

// Render the dashboard
ReactDOM.render(<Dashboard />, document.getElementById('root')); 