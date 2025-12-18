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
                        What GybeLock does, what footage it supports, and common troubleshooting.
                    </p>
                </div>
                <Button variant="primary" onClick={() => navigate('/analyzer')} text="Open Analyzer" />
            </div>

            <div className="flex flex-col gap-3">
                <FaqItem
                    q="What is GybeLock supposed to do?"
                    a={
                        <>
                            GybeLock helps you review <strong>beach-shot</strong> windsurf footage (shore camera, long
                            zoom) by stabilizing the footage, detecting/tracking riders, and letting you click a rider
                            to get a smooth, focused view for review.
                        </>
                    }
                />
                <FaqItem
                    q="What footage is supported?"
                    a={
                        <>
                            GybeLock is designed for <strong>shore / beach-shot</strong> videos (tele/zoom from land).{' '}
                            <strong>MP4</strong> is recommended. GoPro/action-cam POV footage is not supported / not the
                            intended use case.
                        </>
                    }
                />
                <FaqItem
                    q="What is the ingress folder?"
                    a={
                        <>
                            It’s a folder on your computer that GybeLock monitors periodically. When you drop a new
                            video in it, GybeLock will automatically upload it for processing.
                        </>
                    }
                />
                <FaqItem
                    q="Where do processed videos appear?"
                    a={
                        <>
                            They appear in the Analyzer grid as soon as a job is finished. Click a tile to open the
                            player.
                        </>
                    }
                />
                <FaqItem
                    q="Why can’t I open a video yet?"
                    a={
                        <>
                            Only jobs with status “succeeded” can be opened. If it’s processing, wait a moment and
                            refresh.
                        </>
                    }
                />
                <FaqItem
                    q="Why does the player say “VIDEO FILE NOT FOUND”?"
                    a={
                        <>
                            The player opens videos from your local ingress folder. If the file can’t be located, select
                            the correct folder again (Ingress → <strong>Change folder</strong>) and make sure the video
                            still exists in that folder.
                        </>
                    }
                />
                <FaqItem
                    q="I moved or renamed a video. What should I do?"
                    a={
                        <>
                            If the moved/renamed file can’t be found, the player may show{' '}
                            <strong>VIDEO FILE NOT FOUND</strong>. Put the video back into the ingress folder (or
                            expected subfolder) and make sure the correct ingress folder is selected.
                        </>
                    }
                />
                <FaqItem
                    q="Why does upload say “Video too long”?"
                    a={
                        <>
                            There is a maximum supported video length. If your upload fails with “Video too long”, split
                            the recording into shorter clips and upload those.
                        </>
                    }
                />
                <FaqItem
                    q="Why was my upload skipped?"
                    a={
                        <>
                            GybeLock deduplicates videos by checksum. If that exact video was already
                            uploaded/processed, it may be skipped.
                        </>
                    }
                />
                <FaqItem
                    q="Uploads are paused / stuck. What can I do?"
                    a={
                        <>
                            Open <strong>Ingress</strong> and check the error message. If uploads paused after a
                            failure, use <strong>Retry failed</strong>. If you see a quota/free-job message, you’ve hit
                            the current limit for your account.
                        </>
                    }
                />
                <FaqItem
                    q="What formats are supported?"
                    a={<>MP4 is recommended. If you encounter issues, try re-exporting to H.264 MP4.</>}
                />
                <FaqItem
                    q="How do I see all keyboard controls?"
                    a={
                        <>
                            In the Analyzer / Player, click <strong>Shortcuts</strong> to open the keyboard shortcuts
                            modal.
                        </>
                    }
                />
                <FaqItem
                    q="How do I delete my account?"
                    a={
                        <>
                            Open the Analyzer, click <strong>Settings</strong>, then click{' '}
                            <strong>Delete account</strong>.
                        </>
                    }
                />
            </div>
        </div>
    )
}
