import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { ReviewSet } from '../api'
import { api } from '../api'
import { ErrorNote, PageHead, Progress } from '../components'
import { useFetch } from '../hooks'

const CLASSES: Record<string, string[]> = {
  NeuN: ['', 'neuron', 'rejected'],
  '6E10': ['', 'compact', 'diffuse', 'small_plaque'],
  AT8: ['', 'tau_neuron_ring', 'tau_neuron_dense'],
}

export default function Reviews() {
  const sets = useFetch<ReviewSet[]>('/api/reviews')
  const [showLegacy, setShowLegacy] = useState(false)
  const list = (sets.data ?? []).filter((s) => showLegacy || !s.legacy)

  return (
    <>
      <PageHead title="Review & annotate" subtitle="Blinded reference labels: objects are sampled from pipeline outputs, shown as raw crops without group or model score, and labelled by eye. Labels are saved as CSV next to the data.">
        <label className="layer-toggle"><input type="checkbox" checked={showLegacy} onChange={() => setShowLegacy(!showLegacy)} />show study review sets (read-only)</label>
      </PageHead>
      <ErrorNote error={sets.error} />
      <NewReview onCreated={sets.reload} />
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Review sets</h2>
        {list.length ? (
          <table>
            <thead><tr><th>Set</th><th>Stain</th><th style={{ width: 220 }}>Labelled</th><th>Labels</th><th /></tr></thead>
            <tbody>
              {list.map((s) => (
                <tr key={s.name}>
                  <td>{s.title}<div className="muted small">{s.name}{s.legacy ? ' · study set' : ''}</div></td>
                  <td>{s.stain || '—'}</td>
                  <td><Progress done={s.labelled} total={s.items} /><span className="muted small">{s.labelled} / {s.items}</span></td>
                  <td className="small secondary">{Object.entries(s.label_counts).map(([k, v]) => `${k} ${v}`).join(' · ')}</td>
                  <td>{!s.legacy && <Link className="btn small" to={`/reviews/label/${s.name}`}>Open</Link>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="muted">No review sets yet — create one above.</p>}
      </div>
    </>
  )
}

function NewReview({ onCreated }: { onCreated: () => void }) {
  const navigate = useNavigate()
  const [form, setForm] = useState({ name: '', stain: 'AT8', n: 60, strategy: 'random', model_class: '', low: 0.3, high: 0.7, per_group: true, instructions: '' })
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = (key: string, value: unknown) => setForm({ ...form, [key]: value })

  const create = async () => {
    setBusy(true)
    setError(null)
    try {
      const result = await api.post<{ name: string }>('/api/reviews', { ...form, model_class: form.model_class || null })
      onCreated()
      navigate(`/reviews/label/${result.name}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>New review set</h2>
      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="field">Name<input type="text" value={form.name} placeholder="e.g. tau_uncertain_r5" onChange={(e) => set('name', e.target.value)} /></label>
        <label className="field">Stain<select value={form.stain} onChange={(e) => setForm({ ...form, stain: e.target.value, model_class: '' })}>{Object.keys(CLASSES).map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="field">Model class<select value={form.model_class} onChange={(e) => set('model_class', e.target.value)}>{CLASSES[form.stain].map((c) => <option key={c} value={c}>{c || 'any'}</option>)}</select></label>
        <label className="field">Strategy<select value={form.strategy} onChange={(e) => set('strategy', e.target.value)}><option value="random">random</option><option value="uncertain">uncertain probability</option></select></label>
        {form.strategy === 'uncertain' && <>
          <label className="field">p from<input type="number" step="0.05" value={form.low} style={{ width: 70 }} onChange={(e) => set('low', Number(e.target.value))} /></label>
          <label className="field">to<input type="number" step="0.05" value={form.high} style={{ width: 70 }} onChange={(e) => set('high', Number(e.target.value))} /></label>
        </>}
        <label className="field">Items<input type="number" value={form.n} style={{ width: 80 }} onChange={(e) => set('n', Number(e.target.value))} /></label>
        <label className="layer-toggle"><input type="checkbox" checked={form.per_group} onChange={() => set('per_group', !form.per_group)} />balance groups</label>
        <button className="btn primary" disabled={!form.name || busy} onClick={create}>{busy ? 'Sampling…' : 'Create & start'}</button>
      </div>
      <label className="field" style={{ marginTop: 10 }}>Instructions for reviewers<textarea rows={2} value={form.instructions} onChange={(e) => set('instructions', e.target.value)} /></label>
      <ErrorNote error={error} />
    </div>
  )
}
