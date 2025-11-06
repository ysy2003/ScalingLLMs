1. Install gcloud CLI if you don't have it
2. login in google cloud
   ```
   gcloud auth login
   ```
3. add our current project
   ```
   gcloud config set project crack-battery-473522-r9
   ```
4. verify
   ```
   gcloud auth list
   gcloud config list
   ```
5. get dataset from google buckets
   ```
   gcloud storage ls --recursive gs://dataset_design2code
   # you'll see
   ...
    gs://dataset_design2code/:
    gs://dataset_design2code/.gitattributes
    gs://dataset_design2code/10018.html
    gs://dataset_design2code/10018.png
    gs://dataset_design2code/1002.html
    gs://dataset_design2code/1002.png
    gs://dataset_design2code/10032.html
    gs://dataset_design2code/10032.png
    gs://dataset_design2code/1009.html
    gs://dataset_design2code/1009.png
    gs://dataset_design2code/10210.html

   ```
6. Saved in dataURI.txt (no need to run this, just for reference)
   ```
   gcloud storage ls --recursive gs://dataset_design2code/ | findstr /R "\.png$ \.html$" | findstr /V /C:"Design2Code/" | findstr /V /C:".cache/" > dataURI.txt
    ```