import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import type { ReviewData, Row } from '../api'
import { api } from '../api'
import { ErrorNote, PageHead, Progress } from '../components'
import { useFetch, useStored } from '../hooks'
import { useProject } from '../project'

export default function ReviewLabel() {
  const name = useParams()['*'] ?? ''
  const { groupLabel } = useProject()
  const training = name.startsWith('training/')
  const data = useFetch<ReviewData>(`/api/reviews/${name}/items`)
  const [labels, setLabels] = useState<Record<string, string>>({})
  const [index, setIndex] = useState(0)
  const [reviewer, setReviewer] = useStored('stainid.reviewer', '')
  const [confidence, setConfidence] = useState('high')
  const [notes, setNotes] = useState('')
  const [error, setError] = useState<string | null>(null)
  const summary = useFetch<Record<string, Row[]>>(Object.keys(labels).length && data.data && Object.keys(labels).length >= data.data.items.length ? `/api/reviews/${name}/summary` : null)

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
      <PageHead title={data.data?.meta.title ?? name} subtitle={<span><Link to={training ? '/models#train' : '/label'}>← {training ? 'Back to training' : 'All sets'}</Link> · {data.data?.meta.stain} · each picture is {data.data?.meta.fov_um} µm wide; the object to judge is under the blue cross</span>}>
        <label className="field">Reviewer<input type="text" value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="your initials" /></label>
      </PageHead>
      <ErrorNote error={data.error ?? error} />
      {data.data?.meta.instructions && <div className="instructions">{data.data.meta.instructions}</div>}
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
            </div>
            <div className="card">
              <h3>Progress</h3>
              <Progress done={done} total={items.length} />
              <p className="muted small">{done} of {items.length} labelled. Labels are saved as you go, so you can stop and come back any time.
                {training ? ' When you have labelled enough, go back to Models and press Train.' : ' When every item is labelled, the tables below compare your labels with what the model decided and with the diagnostic groups.'}</p>
              {summary.data?.model_class && summary.data.model_class.length > 0 && (
                <Crosstab title="Your labels against what the model decided" rows={summary.data.model_class} header={(k) => MODEL_CLASS[k] ?? k} />
              )}
              {summary.data?.disease_group && summary.data.disease_group.length > 0 && (
                <Crosstab title="Your labels by diagnostic group" rows={summary.data.disease_group} header={groupLabel} />
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

const MODEL_CLASS: Record<string, string> = {
  neuron: 'counted as neuron', rejected: 'rejected', compact: 'compact plaque', diffuse: 'diffuse plaque', small_plaque: 'small plaque',
  tau_neuron_ring: 'tau+ neuron (ring)', tau_neuron_dense: 'tau+ neuron (dense)',
}

function Crosstab({ title, rows, header }: { title: string; rows: Row[]; header: (key: string) => string }) {
  const columns = Object.keys(rows[0]).filter((k) => k !== 'label')
  return (
    <div style={{ marginTop: 12 }}>
      <div className="small secondary" style={{ marginBottom: 4 }}>{title}</div>
      <table className="compact">
        <thead><tr><th>your label</th>{columns.map((c) => <th key={c} className="num">{header(c)}</th>)}</tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={String(r.label)}><td>{String(r.label)}</td>{columns.map((c) => <td key={c} className="num">{String(r[c])}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  )
}
