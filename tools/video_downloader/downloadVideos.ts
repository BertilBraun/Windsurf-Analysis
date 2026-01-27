import admin from 'firebase-admin';
import axios from 'axios';
import fs from 'fs';
import path from 'path';

// --- Configuration ---
// Path to your service account key file
import serviceAccount from './serviceAccountKey.json';

// Initialize Firebase Admin SDK
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: 'gs://gybelock-00.firebasestorage.app'
});

const bucket = admin.storage().bucket();
const storagePath = 'processed'; // The folder in Firebase Storage you want to download from
const localDownloadDir = './downloaded_videos'; // Local directory to save files

// --- Main Function ---
async function downloadFilesFromFolder(folderPath, localDir) {
    console.log(`Starting download from Firebase Storage folder: ${folderPath}`);
    console.log(`Files will be saved to: ${localDir}`);

    // Create the local download directory if it doesn't exist
    if (!fs.existsSync(localDir)) {
        fs.mkdirSync(localDir, { recursive: true });
        console.log(`Created local directory: ${localDir}`);
    }

    try {
        // List all files and sub-folders under the specified path
        // The `prefix` option filters results to objects whose names begin with the prefix.
        const [files] = await bucket.getFiles({ prefix: folderPath });

        if (files.length === 0) {
            console.log(`No files found under '${folderPath}'.`);
            return;
        }

        for (const file of files) {
            // Exclude directories themselves if they appear in the list (often end with '/')
            if (file.name.endsWith('/')) {
                continue;
            }

            const localFilePath = path.join(localDir, file.name);
            const localFileBasedir = path.dirname(localFilePath)
            fs.mkdir(localFileBasedir, { recursive: true }, (err) => {
                if (err) throw err;
            });

            if (fs.existsSync(localFilePath)) continue;

            try {
                // Get the public download URL for the file
                const [url] = await file.getSignedUrl({
                    action: 'read',
                    expires: '03-17-2027', // Set an expiration date far enough in the future
                });

                // Download the file using axios
                const response = await axios({
                    method: 'GET',
                    url: url,
                    responseType: 'stream'
                });

                // Save the downloaded stream to a local file
                const writer = fs.createWriteStream(localFilePath);
                response.data.pipe(writer);

                await new Promise((resolve, reject) => {
                    writer.on('finish', resolve);
                    writer.on('error', reject);
                });

                console.log(`Downloaded: ${file.name} to ${localFilePath}`);
            } catch (downloadError) {
                console.error(`Error downloading ${file.name}:`, downloadError.message);
            }
        }
        console.log('All downloads attempted.');
    } catch (error) {
        console.error('Error listing files from Firebase Storage:', error);
    }
}

// Call the function to start the download process
downloadFilesFromFolder(storagePath, localDownloadDir);
