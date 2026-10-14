import { useMemo, useState } from 'react'
import type { SlideRow, SlidesInfo } from './api'
import { slidePicture } from './api'

/* The slides table editor (guided setup → Slides, and the Register slides step): one row per scan, with a picture of the
   slide where one can be read, and the TMA and stain chosen from lists of what the project knows. A value the file name
   gave clearly is filled in and marked; anything else is left for people to choose, and saving waits until every scan has both. */

const NEW = '__new'
export const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path
const byNumber = (a: string, b: string) => {
  const [x, y] = [Number(a), Number(b)]
  return Number.isNaN(x) || Number.isNaN(y) ? a.localeCompare(b) : x - y
}
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

/** What still stops the slides table from being saved. */
export function slideProblems(rows: SlideRow[], stains: string[]) {
  const needTma = rows.filter((r) => !r.tma?.trim())
  const needStain = rows.filter((r) => !stains.includes(r.stain))
  const twice = rows.filter((r, i) => r.tma && r.stain && rows.findIndex((x) => x.tma === r.tma && x.stain === r.stain) !== i)
  const text = [needTma.length ? `${plural(needTma.length, 'slide still needs', 'slides still need')} a TMA` : '',
    needStain.length ? `${plural(needStain.length, 'slide still needs', 'slides still need')} a stain` : '',
    twice.length ? `${plural(twice.length, 'TMA has', 'TMAs have')} the same stain twice` : ''].filter(Boolean).join(' · ')
  return { needTma, needStain, twice, ok: !needTma.length && !needStain.length && !twice.length, text }
}

/** A TMA as people write it: "LIP-3", "TMA 03" or "3" are TMA 3 when the core names start with LIP-. */
export function cleanTma(text: string, prefix: string) {
  let value = text.trim()
  const strip = (start: string) => { if (start && value.toLowerCase().startsWith(start.toLowerCase())) value = value.slice(start.length).trim() }
  strip(prefix)
  strip(prefix.replace(/[-_ ]+$/, ''))
  value = value.replace(/^tma[-_ #]*/i, '')
  return /^\d+$/.test(value) ? String(Number(value)) : value
}

export function SlideTable({ info, rows, prefix, onChange }: { info: SlidesInfo; rows: SlideRow[]; prefix: string; onChange: (rows: SlideRow[]) => void }) {
  const [added, setAdded] = useState<string[]>([])
  const guessed = useMemo(() => Object.fromEntries(info.rows.map((r) => [r.slide_path, r])), [info.rows])
  const tmas = useMemo(() => [...new Set([...info.tmas, ...added, ...rows.map((r) => r.tma).filter(Boolean)])].sort(byNumber), [info.tmas, added, rows])
  const problems = slideProblems(rows, info.stains)
  const label = (tma: string) => (prefix && prefix !== 'TMA-' ? `${prefix}${tma}` : `TMA ${tma}`)
  const edit = (i: number, change: Partial<SlideRow>) => onChange(rows.map((r, j) => (j === i ? { ...r, ...change } : r)))
  const addTma = (i: number, tma: string) => { if (!added.includes(tma)) setAdded([...added, tma]); edit(i, { tma }) }

  return (
    <div className="table-wrap slide-table" style={{ maxHeight: 520 }}>
      <table>
        <thead><tr><th style={{ width: 222 }}>Slide</th><th>Scan</th><th style={{ width: 190 }}>TMA</th><th style={{ width: 170 }}>Stain</th><th style={{ width: 50 }} /></tr></thead>
        <tbody>{rows.map((r, i) => {
          const first = guessed[r.slide_path]
          const fromName = { tma: Boolean(first?.from_name?.tma && first.tma === r.tma), stain: Boolean(first?.from_name?.stain && first.stain === r.stain) }
          const inName = (first?.options?.tma ?? []).filter((t) => !tmas.includes(t))
          const twice = problems.twice.some((x) => x.tma === r.tma && x.stain === r.stain)
          return (
            <tr key={r.slide_path} className={!r.tma || !info.stains.includes(r.stain) || twice ? 'row-attention' : ''}>
              <td><SlidePicture path={r.slide_path} /></td>
              <td title={r.slide_path}><span className="scan-name">{fileName(r.slide_path)}</span>
                {twice && <div className="small cell-note">{label(r.tma)} has more than one {r.stain} slide</div>}</td>
              <td>
                <TmaChoice value={r.tma} tmas={tmas} inName={inName} label={label} prefix={prefix} name={fileName(r.slide_path)}
                  onChoose={(tma) => edit(i, { tma })} onAdd={(tma) => addTma(i, tma)} />
                {fromName.tma ? <div className="from-name">from the file name</div>
                  : !r.tma && (first?.options?.tma.length ? <div className="small cell-note">the file name has {first.options.tma.join(' or ')}</div>
                    : <div className="small cell-note">choose the TMA</div>)}
              </td>
              <td>
                <select value={info.stains.includes(r.stain) ? r.stain : ''} onChange={(e) => edit(i, { stain: e.target.value })} aria-label={`Stain of ${fileName(r.slide_path)}`}>
                  <option value="">Choose the stain…</option>{info.stains.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
                {fromName.stain ? <div className="from-name">from the file name</div>
                  : !info.stains.includes(r.stain) && <div className="small cell-note">{first?.options?.stain.length ? `the file name has ${first.options.stain.join(' and ')}` : 'choose the stain'}</div>}
              </td>
              <td>{r.new && info.saved && <span className="badge status-running">new</span>}</td>
            </tr>
          )
        })}</tbody>
      </table>
    </div>
  )
}

function TmaChoice({ value, tmas, inName, label, prefix, name, onChoose, onAdd }: {
  value: string; tmas: string[]; inName: string[]; label: (tma: string) => string; prefix: string; name: string
  onChoose: (tma: string) => void; onAdd: (tma: string) => void
}) {
  const [typing, setTyping] = useState(false)
  const [text, setText] = useState('')
  const done = () => { const tma = cleanTma(text, prefix); if (tma) onAdd(tma); setTyping(false); setText('') }
  if (typing) {
    return (
      <div className="row" style={{ gap: 4, flexWrap: 'nowrap' }}>
        <input type="text" value={text} autoFocus placeholder={`e.g. ${prefix && prefix !== 'TMA-' ? prefix : ''}12`} style={{ width: 90 }} aria-label={`New TMA for ${name}`}
          onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') done(); if (e.key === 'Escape') setTyping(false) }} />
        <button className="btn small" disabled={!cleanTma(text, prefix)} onClick={done}>Add</button>
        <button className="link-btn small" onClick={() => setTyping(false)}>cancel</button>
      </div>
    )
  }
  return (
    <select value={value} aria-label={`TMA of ${name}`} onChange={(e) => (e.target.value === NEW ? setTyping(true) : onChoose(e.target.value))}>
      <option value="">Choose the TMA…</option>
      {inName.length > 0 && <optgroup label="In the file name">{inName.map((t) => <option key={t} value={t}>{label(t)}</option>)}</optgroup>}
      {tmas.length > 0 && <optgroup label="TMAs of this study">{tmas.map((t) => <option key={t} value={t}>{label(t)}</option>)}</optgroup>}
      <option value={NEW}>New TMA…</option>
    </select>
  )
}

/** A small picture of the slide (its label and cores) when the scan file keeps one, to recognise it by its label. */
function SlidePicture({ path }: { path: string }) {
  const [missing, setMissing] = useState(false)
  if (missing) return <span className="slide-thumb empty muted small" title="This scan keeps no picture that can be shown quickly">no picture</span>
  return (
    <a href={slidePicture(path, 800)} target="_blank" rel="noreferrer" title="Open the picture larger">
      <img className="slide-thumb" src={slidePicture(path, 440)} alt={`Slide ${fileName(path)}`} loading="lazy" onError={() => setMissing(true)} />
    </a>
  )
}
