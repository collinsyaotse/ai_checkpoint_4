FOR USING THE API TO MAKE A REQUEST RUN THIS IN TERMINAL AND REPLACE "features: []" with the features you want to predict on 

$headers = @{
    "Content-Type" = "application/json"
}

$body = '{"features":[45, 1, 2, 1, 1000,2, 2]}'

Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" -Method POST -Headers $headers -Body $body
