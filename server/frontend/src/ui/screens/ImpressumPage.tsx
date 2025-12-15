import React from 'react'

export const ImpressumPage: React.FC = () => {
    return (
        <div className="max-w-2xl">
            <h1>Impressum</h1>

            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 text-sm text-slate-700 leading-6 space-y-4">
                {/*<div>
                    <div className="text-xs font-semibold text-slate-500">Diensteanbieter</div>
                    <div className="mt-1 font-semibold text-slate-900">Bertil Braun</div>
                    <div>Im Rübländer 19</div>
                    <div>71034 Böblingen</div>
                    <div>Deutschland</div>
                </div>*/}

                <div>
                    <div className="text-xs font-semibold text-slate-500">Kontakt</div>
                    <div className="mt-1">
                        E-Mail:{' '}
                        <a className="text-brand-700 underline" href="mailto:contact@gybelock.de">
                            contact@gybelock.de
                        </a>
                    </div>
                </div>

                {/*<div>
                    <div className="text-xs font-semibold text-slate-500">Verantwortlich für den Inhalt</div>
                    <div className="mt-1">Bertil Braun, Anschrift wie oben.</div>
                </div>*/}
            </div>
        </div>
    )
}
