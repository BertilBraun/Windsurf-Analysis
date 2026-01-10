import { initializeApp } from 'firebase/app'
import { GoogleAuthProvider, getAuth } from 'firebase/auth'
import { initializeFirestore } from 'firebase/firestore'
import { getStorage } from 'firebase/storage'
import { requireEnv } from './env'

const firebaseConfig = {
    apiKey: requireEnv('VITE_FIREBASE_API_KEY'),
    authDomain: requireEnv('VITE_FIREBASE_AUTH_DOMAIN'),
    projectId: requireEnv('VITE_FIREBASE_PROJECT_ID'),
    storageBucket: requireEnv('VITE_FIREBASE_STORAGE_BUCKET'),
    messagingSenderId: requireEnv('VITE_FIREBASE_MESSAGING_SENDER_ID'),
    appId: requireEnv('VITE_FIREBASE_APP_ID'),
}

export const backendUrl = requireEnv('VITE_BACKEND_URL')

export const firebaseApp = initializeApp(firebaseConfig)
export const auth = getAuth(firebaseApp)
export const db = initializeFirestore(firebaseApp, {}, requireEnv('VITE_FIREBASE_DATABASE_ID'))
export const storage = getStorage(firebaseApp)
export const googleProvider = new GoogleAuthProvider()
