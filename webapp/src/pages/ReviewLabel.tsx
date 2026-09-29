import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import type { ReviewData, Row } from '../api'
import { api } from '../api'
import { ErrorNote, GROUP_LABEL, PageHead, Progress } from '../components'
import { useFetch, useStored } from '../hooks'

export default function ReviewLabel() {
  const name = useParams()['*'] ?? ''
  const data = useFetch<ReviewData>(`/api/reviews/${name}/items`)
  const [labels, setLabels] = useState<Record<string, string>>({})
  const [index, setIndex] = useState(0)
  const [reviewer, setReviewer] = useStored('stainid.reviewer', '')
  const [confidence, setConfidence] = useState('high')
  const [notes, setNotes] = useState('')
  const [error, setError] = useState<string | null>(null)
  const summary = useFetch<Row[]>(Object.keys(labels).length && data.data && Object.keys(labels).length >= data.data.items.length ? `/api/reviews/${name}/summary` : null)

  useEffect(() => {
    if (!data.data) return
    const existing = Object.fromEntries(data.data.items.filter((i) => i.label).map((i) => [String(i.review_id), String(i.label)]))
    setLabels(existing)
    const first = data.data.items.findIndex((i) => !i.label)
    setIndex(first === -1 ? 0 : first)
  }, [data.data])

  const items = data.data?.items ?? []
  const item = items[index]
  const options = useMemo(() => data.data?.meta.label_options ?? [], [data.data])

  const save = useCallback(async (label: string) => {
    if (!item) return
    const id = String(item.review_id)
    try {
      await api.post(`/api/reviews/${name}/labels`, { review_id: id, label, confidence, notes, reviewer })
      setLabels((l) => ({ ...l, [id]: label }))
      setNotes('')
      setIndex((i) => Math.min(items.length - 1, i + 1))
    } catch (e) {
      setError((e as Error).message)
    }
  }, [item, name, confidence, notes, reviewer, items.length])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'TEXTAREA' || (e.target as HTMLElement).tagName === 'INPUT') return
      const n = Number(e.key)
      if (n >= 1 && n <= options.length) save(options[n - 1])
      if (e.key === 'ArrowRight') setIndex((i) => Math.min(items.length - 1, i + 1))
      if (e.key === 'ArrowLeft') setIndex((i) => Math.max(0, i - 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [options, save, items.length])

  const done = Object.keys(labels).length

  return (
    <>
      <PageHead title={data.data?.meta.title ?? name} subtitle={<span><Link to="/reviews">Review sets</Link> · {data.data?.meta.stain} · crops are {data.data?.meta.fov_um} µm wide, target at the centre cross</span>}>
        <label className="field">Reviewer<input type="text" value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="your initials" /></label>
      </PageHead>
      <ErrorNote error={data.error ?? error} />
      {item && (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(320px, 560px) 1fr', alignItems: 'start' }}>
          <div className="card">
            <div className="row" style={{ marginBottom: 8 }}>
              <b>{String(item.review_id)}</b><span className="muted small">{index + 1} / {items.length}</span><span className="spacer" />
              {labels[String(item.review_id)] && <span className="badge">labelled: {labels[String(item.review_id)]}</span>}
            </div>
            <div style={{ position: 'relative' }}>
              <img src={`/api/reviews/${name}/image/${item.review_id}.jpg`} alt="review crop" style={{ width: '100%', display: 'block', borderRadius: 8, imageRendering: 'auto' }} />
              <svg viewBox="0 0 100 100" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                <path d="M50 44 V47 M50 53 V56 M44 50 H47 M53 50 H56" stroke="#00e5ff" strokeWidth={0.8} />
              </svg>
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn" onClick={() => setIndex(Math.max(0, index - 1))}>← Prev</button>
              <button className="btn" onClick={() => setIndex(Math.min(items.length - 1, index + 1))}>Next →</button>
              <span className="spacer" />
              <button className="btn small" onClick={() => setIndex(Math.max(0, items.findIndex((i) => !labels[String(i.review_id)])))}>Next unlabelled</button>
            </div>
          </div>
          <div className="grid">
            <div className="card">
              <h3>Label</h3>
              <div className="grid" style={{ gap: 6 }}>
                {options.map((option, i) => (
                  <button key={option} className={`btn${labels[String(item.review_id)] === option ? ' primary' : ''}`} style={{ textAlign: 'left' }} onClick={() => save(option)}>
                    <span className="kbd">{i + 1}</span> {option.replaceAll('_', ' ')}
                  </button>
                ))}
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <label className="field">Confidence<select value={confidence} onChange={(e) => setConfidence(e.target.value)}><option>high</option><option>moderate</option><option>low</option></select></label>
                <label className="field" style={{ flex: 1 }}>Notes<input type="text" value={notes} onChange={(e) => setNotes(e.target.value)} /></label>
              </div>
              <p className="muted small">Keys: <span className="kbd">1</span>–<span className="kbd">{options.length}</span> label, <span className="kbd">←</span>/<span className="kbd">→</span> navigate.</p>
              {data.data?.meta.instructions && <p className="secondary small">{data.data.meta.instructions}</p>}
            </div>
            <div className="card">
              <h3>Progress</h3>
              <Progress done={done} total={items.length} />
              <p className="muted small">{done} of {items.length} labelled. Group and model score stay hidden until every item is labelled.</p>
              {summary.data && summary.data.length > 0 && (
                <table>
                  <thead><tr><th>label</th>{Object.keys(summary.data[0]).filter((k) => k !== 'label').map((g) => <th key={g} className="num">{GROUP_LABEL[g] ?? g}</th>)}</tr></thead>
                  <tbody>{summary.data.map((r) => (
                    <tr key={String(r.label)}><td>{String(r.label)}</td>{Object.entries(r).filter(([k]) => k !== 'label').map(([k, v]) => <td key={k} className="num">{String(v)}</td>)}</tr>
                  ))}</tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
