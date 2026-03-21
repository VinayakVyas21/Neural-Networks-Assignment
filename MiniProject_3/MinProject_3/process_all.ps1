$asanas = Get-ChildItem -Path Dataset -Directory
$output_file = "processed_data_new.pkl"

if (Test-Path $output_file) { Remove-Item $output_file }

foreach ($asana in $asanas) {
    $asana_name = $asana.Name
    $qualities = Get-ChildItem -Path $asana.FullName -Directory
    
    foreach ($quality in $qualities) {
        $quality_name = $quality.Name
        $videos = Get-ChildItem -Path $quality.FullName -Include *.mp4, *.avi, *.mov -File | Select-Object -First 2
        
        foreach ($video in $videos) {
            Write-Host "Processing $asana_name - $quality_name - $($video.Name)"
            .\yoga_venv\Scripts\python.exe process_single_video.py $video.FullName $asana_name $quality_name $output_file
        }
    }
}
