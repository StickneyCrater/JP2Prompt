def get_web_ui_html():
    """WebUI用のHTML文字列を返す"""
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Generation Settings</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            textarea { height: 100px; resize: vertical; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
            button:hover { background-color: #0056b3; }
            .btn-secondary { background-color: #6c757d; }
            .btn-secondary:hover { background-color: #545b62; }
            .result { margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; display: none; }
            .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .config-section { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }
            .form-row { display: flex; gap: 15px; }
            .form-row .form-group { flex: 1; }
            .image-result { margin-top: 20px; }
            .image-result img { max-width: 100%; height: auto; border-radius: 4px; margin-bottom: 10px; }
            .loading-gif { max-width: 200px; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Image Generation Settings</h1>
            
            <form id="imageForm">
                <div class="form-group">
                    <label for="japanesePrompt">日本語プロンプト:</label>
                    <textarea id="japanesePrompt" name="japanese_prompt" placeholder="生成したい画像の説明を日本語で入力してください" required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="negativePrompt">ネガティブプロンプト:</label>
                    <textarea id="negativePrompt" name="negative_prompt" placeholder="避けたい要素を入力"></textarea>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="width">Width:</label>
                        <input type="number" id="width" name="width" value="512" min="64" max="2048" step="64">
                    </div>
                    <div class="form-group">
                        <label for="height">Height:</label>
                        <input type="number" id="height" name="height" value="512" min="64" max="2048" step="64">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="cfgScale">CFG Scale:</label>
                        <input type="number" id="cfgScale" name="cfg_scale" value="7.0" min="1" max="30" step="0.5">
                    </div>
                    <div class="form-group">
                        <label for="steps">Steps:</label>
                        <input type="number" id="steps" name="steps" value="20" min="1" max="100">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="batchSize">Batch Size:</label>
                        <input type="number" id="batchSize" name="batch_size" value="1" min="1" max="8">
                    </div>
                    <div class="form-group">
                        <label for="batchCount">Batch Count:</label>
                        <input type="number" id="batchCount" name="batch_count" value="1" min="1" max="10">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="selectedTextEncoder">Text Encoder:</label>
                        <select id="selectedTextEncoder" name="selected_text_encoder">
                            <option value="">Default</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="selectedUnet">UNET:</label>
                        <select id="selectedUnet" name="selected_unet">
                            <option value="">Default</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="dynamicPrompts" name="dynamic_prompts"> Dynamic Prompts ON
                    </label>
                </div>
                
                <button type="submit">Generate Image</button>
                <button type="button" class="btn-secondary" onclick="loadCurrentConfig()">Load Current Settings</button>
            </form>
            
            <div id="result" class="result"></div>
            <div id="imageResult" class="image-result"></div>
            
            <div class="config-section">
                <h2>Model Configuration</h2>
                <form id="configForm">
                    <div class="form-group">
                        <label for="selectedModel">Model:</label>
                        <select id="selectedModel" name="sd_model_checkpoint">
                            <option value="">Loading...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="selectedVae">VAE:</label>
                        <select id="selectedVae" name="sd_vae">
                            <option value="Automatic">Automatic</option>
                            <option value="None">None</option>
                        </select>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="defaultWidth">Default Width:</label>
                            <input type="number" id="defaultWidth" name="default_width" value="512" min="64" max="2048" step="64">
                        </div>
                        <div class="form-group">
                            <label for="defaultHeight">Default Height:</label>
                            <input type="number" id="defaultHeight" name="default_height" value="512" min="64" max="2048" step="64">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="defaultPrompt">Default Prompt (プロンプトの前に自動付与):</label>
                        <textarea id="defaultPrompt" name="default_prompt" placeholder="masterpiece, best quality, highly detailed, "></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="defaultNegativePrompt">Default Negative Prompt (ネガティブプロンプトの前に自動付与):</label>
                        <textarea id="defaultNegativePrompt" name="default_negative_prompt" placeholder="lowres, bad anatomy, bad hands, text, error, "></textarea>
                    </div>
                    
                    <button type="button" onclick="updateConfig()">Update Configuration</button>
                    <button type="button" class="btn-secondary" onclick="resetConfig()">Reset to Defaults</button>
                    <button type="button" class="btn-secondary" onclick="loadConfigHistory()">Show History</button>
                </form>
            </div>
        </div>
        
        <script>
            // 画像生成フォーム送信
            document.getElementById('imageForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                // チェックボックスの処理
                data.dynamic_prompts = document.getElementById('dynamicPrompts').checked;
                
                // 数値型に変換
                ['width', 'height', 'steps', 'batch_size', 'batch_count'].forEach(key => {
                    data[key] = parseInt(data[key]);
                });
                data.cfg_scale = parseFloat(data.cfg_scale);
                
                // ローディングGIFを表示
                showLoadingImage();
                showResult('Generating image...', 'success');
                
                try {
                    const response = await fetch('/get_image', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showResult('Image generated successfully!', 'success');
                        if (result.images && result.images.length > 0) {
                            displayImages(result.images);
                        }
                    } else {
                        showResult('Error: ' + result.detail, 'error');
                        showErrorImage();
                    }
                } catch (error) {
                    showResult('Network error: ' + error.message, 'error');
                    showErrorImage();
                }
            });
            
            function showResult(message, type) {
                const resultDiv = document.getElementById('result');
                resultDiv.textContent = message;
                resultDiv.className = 'result ' + type;
                resultDiv.style.display = 'block';
            }
            
            function showLoadingImage() {
                const imageDiv = document.getElementById('imageResult');
                imageDiv.innerHTML = '<img src="/static/animated.gif" alt="Loading..." class="loading-gif">';
            }
            
            function showErrorImage() {
                const imageDiv = document.getElementById('imageResult');
                imageDiv.innerHTML = '<img src="/static/err.gif" alt="Error" class="loading-gif">';
            }
            
            function displayImages(images) {
                const imageDiv = document.getElementById('imageResult');
                imageDiv.innerHTML = '';
                
                images.forEach((imageBase64, index) => {
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + imageBase64;
                    img.alt = 'Generated image ' + (index + 1);
                    imageDiv.appendChild(img);
                });
            }
            
            async function updateConfig() {
                const formData = new FormData(document.getElementById('configForm'));
                const data = Object.fromEntries(formData.entries());
                
                // 数値型に変換
                ['default_width', 'default_height'].forEach(key => {
                    if (data[key]) data[key] = parseInt(data[key]);
                });
                
                try {
                    const response = await fetch('/config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showResult('Configuration updated successfully!', 'success');
                    } else {
                        showResult('Error: ' + result.detail, 'error');
                    }
                } catch (error) {
                    showResult('Network error: ' + error.message, 'error');
                }
            }
            
            async function loadCurrentConfig() {
                try {
                    const response = await fetch('/config');
                    const result = await response.json();
                    
                    if (response.ok) {
                        const config = result.current_config;
                        document.getElementById('width').value = config.default_width || 512;
                        document.getElementById('height').value = config.default_height || 512;
                        document.getElementById('cfgScale').value = config.default_cfg_scale || 7.0;
                        document.getElementById('steps').value = config.default_steps || 20;
                        document.getElementById('batchSize').value = config.default_batch_size || 1;
                        document.getElementById('batchCount').value = config.default_batch_count || 1;
                        document.getElementById('dynamicPrompts').checked = config.dynamic_prompts_enabled || false;
                        
                        showResult('Current settings loaded!', 'success');
                    }
                } catch (error) {
                    showResult('Error loading config: ' + error.message, 'error');
                }
            }
            
            async function loadConfigHistory() {
                try {
                    const response = await fetch('/config');
                    const result = await response.json();
                    
                    if (response.ok && result.config_history.length > 0) {
                        const historyText = result.config_history
                            .map(h => h.timestamp + ': ' + JSON.stringify(h.config, null, 2))
                            .join('\\n\\n');
                        alert('Configuration History:\\n\\n' + historyText);
                    } else {
                        showResult('No configuration history available.', 'success');
                    }
                } catch (error) {
                    showResult('Error loading history: ' + error.message, 'error');
                }
            }
            
            async function resetConfig() {
                if (!confirm('Are you sure you want to reset all settings to default values?')) {
                    return;
                }
                
                try {
                    const response = await fetch('/config/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showResult('Configuration reset to defaults!', 'success');
                        // ページをリロードして最新の設定を反映
                        setTimeout(() => location.reload(), 1000);
                    } else {
                        showResult('Error: ' + result.detail, 'error');
                    }
                } catch (error) {
                    showResult('Network error: ' + error.message, 'error');
                }
            }
            
            window.addEventListener('load', async function() {
                try {
                    const response = await fetch('/config');
                    const result = await response.json();
                    
                    if (response.ok) {
                        // モデル一覧を設定
                        const modelSelect = document.getElementById('selectedModel');
                        modelSelect.innerHTML = '<option value="">Select Model</option>';
                        result.available_models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            modelSelect.appendChild(option);
                        });
                        
                        // モジュール一覧を設定（Text Encoder, UNET用）
                        const textEncoderSelect = document.getElementById('selectedTextEncoder');
                        const unetSelect = document.getElementById('selectedUnet');
                        
                        result.available_modules.forEach(module => {
                            // Text Encoderオプションを追加
                            const textEncoderOption = document.createElement('option');
                            textEncoderOption.value = module.filename;
                            textEncoderOption.textContent = module.model_name;
                            textEncoderSelect.appendChild(textEncoderOption);
                            
                            // UNETオプションを追加
                            const unetOption = document.createElement('option');
                            unetOption.value = module.filename;
                            unetOption.textContent = module.model_name;
                            unetSelect.appendChild(unetOption);
                        });
                        
                        if (result.current_config.sd_model_checkpoint) {
                            modelSelect.value = result.current_config.sd_model_checkpoint;
                        }
                        
                        // フォームに現在の設定を反映
                        const config = result.current_config;
                        document.getElementById('defaultWidth').value = config.default_width || 512;
                        document.getElementById('defaultHeight').value = config.default_height || 512;
                        document.getElementById('selectedVae').value = config.sd_vae || 'Automatic';
                        document.getElementById('defaultPrompt').value = config.default_prompt || '';
                        document.getElementById('defaultNegativePrompt').value = config.default_negative_prompt || '';
                    }
                } catch (error) {
                    console.error('Error loading initial config:', error);
                }
            });
        </script>
    </body>
    </html>
    """