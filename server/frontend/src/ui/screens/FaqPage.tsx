import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/Button'

const FaqItem: React.FC<{ q: string; a: React.ReactNode }> = ({ q, a }) => {
    return (
        <details className="rounded-xl border border-slate-200 bg-white p-4">
            <summary className="cursor-pointer text-sm font-semibold text-slate-900">{q}</summary>
            <div className="mt-2 text-sm text-slate-600 leading-6">{a}</div>
        </details>
    )
}

export const FaqPage: React.FC = () => {
    const navigate = useNavigate()
    return (
        <div className="flex flex-col gap-6">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <h1>FAQ</h1>
                    <p className="text-sm text-slate-600">
                        The basics of using GybeLock: ingress folder, uploads, and viewing results.
                    </p>
                </div>
                <Button variant="primary" onClick={() => navigate('/analyzer')} text="Open Analyzer" />
            </div>

            <div className="flex flex-col gap-3">
                <FaqItem
                    q="What is the ingress folder?"
                    a={
                        <>
                            It’s a folder on your computer that GybeLock monitors periodically. When you drop a new video
                            in it, GybeLock will automatically upload it for processing.
                        </>
                    }
                />
                <FaqItem
                    q="Where do processed videos appear?"
                    a={<>They appear in the Analyzer grid as soon as a job is finished. Click a tile to open the player.</>}
                />
                <FaqItem
                    q="Why can’t I open a video yet?"
                    a={<>Only jobs with status “succeeded” can be opened. If it’s processing, wait a moment and refresh.</>}
                />
                <FaqItem
                    q="What formats are supported?"
                    a={<>MP4 is recommended. If you encounter issues, try re-exporting to H.264 MP4.</>}
                />
            </div>
        </div>
    )
}


