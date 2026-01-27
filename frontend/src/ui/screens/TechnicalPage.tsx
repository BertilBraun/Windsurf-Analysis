/**
 * @module TechnicalPage
 * @description Provides a screen for displaying technical documentation and repository links.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import { Button } from '../components/Button'
import { Heading, Text, TextStack } from '../components/Typography'

const TECHNICAL_DOC_URL = '/TECHNICAL.md'
const GITHUB_REPO_URL = 'https://github.com/BertilBraun/Windsurf-Analysis/tree/production'

/**
 * A screen component that fetches and renders the project's technical documentation.
 *
 * It retrieves the `TECHNICAL.md` file from the server and displays it using
 * a custom-styled Markdown renderer, alongside links to the source repository.
 */
export const TechnicalPage: React.FC = () => {
    const { t } = useTranslation()
    const [markdown, setMarkdown] = React.useState<string>('')
    const [loading, setLoading] = React.useState<boolean>(true)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        let cancelled = false
        setLoading(true)
        setError(null)

        void (async () => {
            try {
                const res = await fetch(TECHNICAL_DOC_URL, { cache: 'no-store' })
                if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
                const text = await res.text()
                if (cancelled) return
                setMarkdown(text)
            } catch (e) {
                if (cancelled) return
                setError(e instanceof Error ? e.message : String(e))
            } finally {
                if (cancelled) return
                setLoading(false)
            }
        })()

        return () => {
            cancelled = true
        }
    }, [])

    return (
        <div className="flex flex-col gap-6">
            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div className="max-w-3xl">
                        <Heading level={1}>{t('screens.technical.title')}</Heading>
                        <TextStack className="mt-3">
                            <Text>{t('screens.technical.subtitle')}</Text>
                        </TextStack>
                    </div>
                    <div className="flex gap-2">
                        <Button
                            variant="outline"
                            size="md"
                            onClick={() => window.open(GITHUB_REPO_URL, '_blank', 'noopener,noreferrer')}
                            text={t('screens.technical.actions.github')}
                        />
                    </div>
                </div>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                {loading && <div className="text-sm text-slate-600">{t('screens.technical.loading')}</div>}
                {error && (
                    <div className="text-sm text-red-600">
                        {t('screens.technical.error', { message: error })}
                    </div>
                )}
                {!loading && !error && (
                    <div className="max-w-4xl">
                        <ReactMarkdown
                            components={{
                                h1: ({ children }) => (
                                    <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-8 mb-3">
                                        {children}
                                    </h1>
                                ),
                                h2: ({ children }) => (
                                    <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 mt-8 mb-3">
                                        {children}
                                    </h2>
                                ),
                                h3: ({ children }) => (
                                    <h3 className="text-lg font-semibold text-slate-900 mt-6 mb-2">{children}</h3>
                                ),
                                p: ({ children }) => <p className="text-slate-700 leading-relaxed my-3">{children}</p>,
                                ul: ({ children }) => <ul className="list-disc pl-6 space-y-1 my-3">{children}</ul>,
                                ol: ({ children }) => (
                                    <ol className="list-decimal pl-6 space-y-1 my-3">{children}</ol>
                                ),
                                li: ({ children }) => <li className="text-slate-700 leading-relaxed">{children}</li>,
                                a: ({ href, children }) => (
                                    <a
                                        href={href}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="text-brand-700 underline underline-offset-4 hover:text-brand-800"
                                    >
                                        {children}
                                    </a>
                                ),
                                code: ({ className, children, ...props }) => {
                                    const isBlock = typeof className === 'string' && className.includes('language-')
                                    if (!isBlock) {
                                        return (
                                            <code
                                                className="rounded bg-slate-100 px-1.5 py-0.5 text-[0.9em] font-mono text-slate-900"
                                                {...props}
                                            >
                                                {children}
                                            </code>
                                        )
                                    }
                                    return (
                                        <code className="text-[0.9em] font-mono text-slate-100" {...props}>
                                            {children}
                                        </code>
                                    )
                                },
                                pre: ({ children }) => (
                                    <pre className="my-4 overflow-auto rounded-lg bg-slate-900 p-4 text-slate-100">
                                        {children}
                                    </pre>
                                ),
                                hr: () => <hr className="my-6 border-slate-200" />,
                            }}
                        >
                            {markdown}
                        </ReactMarkdown>
                    </div>
                )}
            </section>
        </div>
    )
}
