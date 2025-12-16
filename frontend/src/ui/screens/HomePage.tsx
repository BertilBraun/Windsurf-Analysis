import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/Button'

export const HomePage: React.FC = () => {
    const navigate = useNavigate()
    return (
        <div className="flex flex-col gap-10">
            <section className="rounded-2xl border border-slate-200 bg-gradient-to-b from-white to-slate-50 p-6 sm:p-10">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                    <div className="max-w-2xl">
                        <div className="text-xs font-semibold text-brand-700 mb-2">GybeLock</div>
                        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">See your windsurfing clearly.</h1>
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            <strong>GybeLock</strong> turns shaky, long-zoom beach videos into smooth, centered clips that
                            are easy to watch, analyze, and learn from.
                        </p>
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            From speed runs to wave riding, jumps, and gybes — GybeLock keeps the rider where your focus
                            should be.
                        </p>

                        <div className="mt-5 flex flex-col sm:flex-row gap-3 sm:items-center">
                            <Button variant="primary" onClick={() => navigate('/analyzer')} text="Get started for free" />
                            <div className="text-sm text-slate-700">
                                Analyze up to <span className="font-semibold">5 videos</span> free. No credit card.
                            </div>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden">
                        <div className="px-4 py-3 border-b border-slate-200 flex items-center justify-between">
                            <div className="text-sm font-semibold">Before / After</div>
                            <div className="text-xs text-slate-500">Add your screenshots to enable the comparison</div>
                        </div>
                        <div className="grid grid-cols-2">
                            <ComparisonTile label="Before (raw)" src="/marketing/before.jpg" />
                            <ComparisonTile label="After (GybeLock)" src="/marketing/after.jpg" />
                        </div>
                    </div>
                </div>
            </section>

            <Section title="The problem">
                <p>Filming windsurfing from the beach isn’t easy.</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>Long zoom amplifies every small camera movement</li>
                    <li>Riders constantly move across the frame</li>
                    <li>Key moments happen fast — and get lost just as fast</li>
                </ul>
                <p className="mt-3">The result: footage that’s hard to watch, and even harder to learn from.</p>
            </Section>

            <Section title="What GybeLock does">
                <p>GybeLock automatically:</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <strong>Stabilizes</strong> shaky beach footage
                    </li>
                    <li>
                        <strong>Locks onto</strong> a selected rider
                    </li>
                    <li>
                        <strong>Keeps them centered</strong> in the frame
                    </li>
                    <li>
                        <strong>Smoothly follows</strong> fast and dynamic movements
                    </li>
                </ul>
                <div className="mt-3 text-slate-600">
                    Whether you’re looking at speed runs, wave riding, big conditions, jumps and landings, or technical
                    transitions — GybeLock helps you actually <em>see</em> what’s happening.
                </div>
            </Section>

            <Section title="Built for real windsurf footage">
                <p>GybeLock is designed specifically for:</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>Long-distance filming from shore</li>
                    <li>Multiple riders in the same video</li>
                    <li>Challenging conditions like spray, chop, and partial occlusions</li>
                </ul>
                <p className="mt-3">You choose who to follow. GybeLock handles the motion.</p>
            </Section>

            <Section title="How it works">
                <ol className="list-decimal pl-5 space-y-2">
                    <li>
                        <strong>Add your videos</strong>
                        <div className="text-sm text-slate-600">Upload them or drop them into a folder.</div>
                    </li>
                    <li>
                        <strong>GybeLock analyzes the footage</strong>
                        <div className="text-sm text-slate-600">
                            The camera motion is stabilized and riders are detected.
                        </div>
                    </li>
                    <li>
                        <strong>Select a rider</strong>
                        <div className="text-sm text-slate-600">Click on the rider you want to focus on.</div>
                    </li>
                    <li>
                        <strong>Watch the locked view</strong>
                        <div className="text-sm text-slate-600">
                            Smooth, centered footage — ready for analysis. No editing experience required.
                        </div>
                    </li>
                </ol>
            </Section>

            <Section title="Who it’s for">
                <ul className="list-disc pl-5 space-y-1">
                    <li>Windsurfers who want to understand their riding</li>
                    <li>Coaches reviewing sessions</li>
                    <li>Friends filming from the beach</li>
                    <li>Anyone tired of shaky zoom videos</li>
                </ul>
                <p className="mt-3">Beginner or advanced — clear footage helps everyone improve.</p>
            </Section>

            <Section title="Why GybeLock">
                <p>Because better feedback starts with better footage.</p>
                <p className="mt-2">GybeLock removes distraction, so you can focus on technique, timing, and flow.</p>
            </Section>

            <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8 flex flex-col sm:flex-row gap-4 sm:items-center">
                <div className="flex-1">
                    <div className="text-xs font-semibold text-brand-700">Get started</div>
                    <div className="mt-1 text-lg font-semibold text-slate-900">5 videos are free for everyone</div>
                    <div className="mt-1 text-sm text-slate-700">No credit card required. See the difference in minutes.</div>
                </div>
                <Button variant="primary" onClick={() => navigate('/analyzer')} text="Get started for free" />
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <div className="flex items-start justify-between gap-4">
                    <div>
                        <div className="text-sm font-semibold">Make the homepage more visual</div>
                        <div className="mt-1 text-sm text-slate-600">
                            Add images to <code className="text-xs bg-slate-100 px-1 py-0.5 rounded">server/frontend/public/marketing</code>{' '}
                            using the filenames below.
                        </div>
                    </div>
                </div>
                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                    <AssetSuggestion
                        title="Before"
                        filename="/marketing/before.jpg"
                        desc="One raw beach zoom frame with shake + rider near an edge."
                    />
                    <AssetSuggestion
                        title="After"
                        filename="/marketing/after.jpg"
                        desc="Same moment after GybeLock: stabilized + rider centered."
                    />
                    <AssetSuggestion
                        title="Overlay"
                        filename="/marketing/overlay.jpg"
                        desc="A player screenshot showing overlay/tracking to communicate “analysis”."
                    />
                </div>
            </section>
        </div>
    )
}

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => {
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
            <h2 className="mt-0">{title}</h2>
            <div className="text-sm text-slate-600 leading-6 space-y-3">{children}</div>
        </section>
    )
}

const ComparisonTile: React.FC<{ label: string; src: string }> = ({ label, src }) => {
    return (
        <div className="relative aspect-video bg-slate-100 border-r border-slate-200 last:border-r-0">
            <img
                src={src}
                alt={label}
                className="absolute inset-0 w-full h-full object-cover"
                onError={e => {
                    ;(e.currentTarget as HTMLImageElement).style.display = 'none'
                }}
            />
            <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-xs text-slate-500 px-3 py-1 rounded-full bg-white/80 border border-slate-200">
                    {label}
                </div>
            </div>
        </div>
    )
}

const AssetSuggestion: React.FC<{ title: string; filename: string; desc: string }> = ({ title, filename, desc }) => {
    return (
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="text-sm font-semibold">{title}</div>
            <div className="mt-1 text-xs text-slate-600">
                <code className="bg-white border border-slate-200 px-1 py-0.5 rounded">{filename}</code>
            </div>
            <div className="mt-2 text-sm text-slate-600 leading-6">{desc}</div>
        </div>
    )
}


