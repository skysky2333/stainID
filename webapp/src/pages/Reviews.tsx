import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { ReviewSet } from '../api'
import { api } from '../api'
import { ErrorNote, HelpBox, PageHead, Progress, Term, withError } from '../components'
import { useFetch } from '../hooks'

const CLASSES: Record<string, [string, string][]> = {
  NeuN: [['', 'any'], ['neuron', 'counted as neuron'], ['rejected', 'rejected candidates']],
  '6E10': [['', 'any'], ['compact', 'compact plaques'], ['diffuse', 'diffuse plaques'], ['small_plaque', 'small plaques']],
  AT8: [['', 'any'], ['tau_neuron_ring', 'tau+ neurons (ring)'], ['tau_neuron_dense', 'tau+ neurons (dense body)']],
}

export default function Reviews() {
  const sets = useFetch<ReviewSet[]>('/api/reviews')
  const [showLegacy, setShowLegacy] = useState(false)
  const all = sets.data ?? []

  return (
    <>
      <PageHead title="Label & check" subtitle="Look at objects one by one and say what they are — blind to diagnosis and to what the model decided." />
      <HelpBox id="label">
        <ul>
          <li><b><Term term="training set">Training sets</Term></b> contain candidate objects (real and false) for teaching a model. Create them on the <Link to="/models">Models</Link> page.</li>
          <li><b><Term term="check set">Check sets</Term></b> contain objects the pipeline already detected. Labelling them tells you how accurate the current results are.</li>
          <li>While you label, the diagnosis, donor and the model’s answer stay hidden (<Term>blinded</Term>). Labels are saved instantly as CSV in the project folder.</li>
        </ul>
      </HelpBox>
      <ErrorNote error={sets.error} />
      <SetTable title="Training sets" sets={all.filter((s) => s.purpose === 'training')} empty={<>None yet — create one on the <Link to="/models">Models</Link> page.</>} />
      <SetTable title="Check sets" sets={all.filter((s) => s.purpose === 'check')} empty="None yet — create one below." />
      <NewCheckSet onCreated={sets.reload} />
      <label className="layer-toggle" style={{ marginTop: 12 }}><input type="checkbox" checked={showLegacy} onChange={() => setShowLegacy(!showLegacy)} />show older study review sets (read-only)</label>
      {showLegacy && <SetTable title="Older study sets" sets={all.filter((s) => s.purpose === 'legacy')} empty="None." readOnly />}
    </>
  )
}

function SetTable({ title, sets, empty, readOnly = false }: { title: string; sets: ReviewSet[]; empty: React.ReactNode; readOnly?: boolean }) {
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h2>{title}</h2>
      {sets.length ? (
        <table>
          <thead><tr><th>Set</th><th>Stain</th><th style={{ width: 220 }}>Labelled</th><th>Labels so far</th><th /></tr></thead>
          <tbody>
            {sets.map((s) => (
              <tr key={s.name}>
                <td>{s.title}<div className="muted small">{s.name}</div></td>
                <td>{s.stain || '—'}</td>
                <td><Progress done={s.labelled} total={s.items} /><span className="muted small">{s.labelled} of {s.items}</span></td>
                <td className="small secondary">{Object.entries(s.label_counts).map(([k, v]) => `${k.replaceAll('_', ' ')}: ${v}`).join(' · ')}</td>
                <td>{!readOnly && <Link className="btn small primary" to={`/label/${s.name}`}>{s.labelled < s.items ? 'Label' : 'Open'}</Link>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted small">{empty}</p>}
    </div>
  )
}

function NewCheckSet({ onCreated }: { onCreated: () => void }) {
  const navigate = useNavigate()
  const [form, setForm] = useState({ name: '', stain: 'AT8', n: 60, strategy: 'random', model_class: '', low: 0.3, high: 0.7, per_group: true, instructions: '' })
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = (key: string, value: unknown) => setForm({ ...form, [key]: value })

  const create = async () => {
    setBusy(true)
    const result = await withError(() => api.post<{ name: string }>('/api/reviews', { ...form, model_class: form.model_class || null }), setError)
    setBusy(false)
    if (result) { onCreated(); navigate(`/label/${result.name}`) }
  }

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h2>New check set</h2>
      <p className="muted small">Picks detected objects at random (optionally only one kind, or only ones the model was unsure about), spread evenly over the diagnostic groups.</p>
      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="field">Name<input type="text" value={form.name} placeholder="e.g. tau_check_1" onChange={(e) => set('name', e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
        <label className="field">Stain<select value={form.stain} onChange={(e) => setForm({ ...form, stain: e.target.value, model_class: '' })}>{Object.keys(CLASSES).map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="field">Which objects<select value={form.model_class} onChange={(e) => set('model_class', e.target.value)}>{CLASSES[form.stain].map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
        <label className="field">Pick<select value={form.strategy} onChange={(e) => set('strategy', e.target.value)}><option value="random">at random</option><option value="uncertain">where the model was unsure</option></select></label>
        {form.strategy === 'uncertain' && <>
          <label className="field">probability from<input type="number" step="0.05" value={form.low} style={{ width: 70 }} onChange={(e) => set('low', Number(e.target.value))} /></label>
          <label className="field">to<input type="number" step="0.05" value={form.high} style={{ width: 70 }} onChange={(e) => set('high', Number(e.target.value))} /></label>
        </>}
        <label className="field">How many<input type="number" value={form.n} style={{ width: 80 }} onChange={(e) => set('n', Number(e.target.value))} /></label>
        <label className="layer-toggle"><input type="checkbox" checked={form.per_group} onChange={() => set('per_group', !form.per_group)} />same number per group</label>
        <button className="btn primary" disabled={!form.name || busy} onClick={create}>{busy ? 'Picking objects…' : 'Create and start labelling'}</button>
      </div>
      <label className="field" style={{ marginTop: 10 }}>Instructions shown to whoever labels (optional)<textarea rows={2} value={form.instructions} onChange={(e) => set('instructions', e.target.value)} /></label>
      <ErrorNote error={error} />
    </div>
  )
}
