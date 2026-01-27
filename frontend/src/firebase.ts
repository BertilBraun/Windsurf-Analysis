/**
 * @file Firebase configuration and service initialization.
 * This module initializes the Firebase application and provides access to
 * Auth, Firestore, and Storage services.
 */

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

/**
 * The base URL for the backend API services.
 */
export const backendUrl = requireEnv('VITE_BACKEND_URL')

/**
 * The initialized Firebase application instance.
 */
export const firebaseApp = initializeApp(firebaseConfig)

/**
 * Firebase Authentication service instance.
 */
export const auth = getAuth(firebaseApp)

/**
 * Firestore database instance, initialized with a specific database ID from environment variables.
 */
export const db = initializeFirestore(firebaseApp, {}, requireEnv('VITE_FIREBASE_DATABASE_ID'))

/**
 * Firebase Cloud Storage service instance.
 */
export const storage = getStorage(firebaseApp)

/**
 * Google Authentication provider instance for use with Firebase Auth.
 */
export const googleProvider = new GoogleAuthProvider()
