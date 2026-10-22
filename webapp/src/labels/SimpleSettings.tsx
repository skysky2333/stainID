/* The simple classes of "Fix the field (simple)" (labels/simple.py): Settings → Simple classes (what people choose from,
   per stain: names, descriptions, keys, colours, which of detection's calls a candidate starts in each from) and the
   Models page's "Simple labels in training" (how each simple class trains: as one of the detailed classes, or not at
   all). Simple labels stay as they are in the label log; only training reads them through this mapping, and each
   model records the mapping it used. */
import { useEffect, useState } from 'react'
import type { SimpleClassDef, SimpleSettings, TrainedModel } from '../api'
import { api } from '../api'
import { ErrorNote, Loaded, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced } from '../setup/parts'
import { PALETTE } from '../palette'

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

function StainTabs({ stains, shown, onPick }: { stains: string[]; shown: string; onPick: (s: string) => void }) {
  return stains.length > 1 ? <div className="segmented">{stains.map((s) => (
    <button key={s} className={`btn small${s === shown ? ' chosen' : ''}`} aria-pressed={s === shown} onClick={() => onPick(s)}>{s}</button>
  ))}</div> : null
}

/** Models page: each simple class -> the detailed class it trains as (or unused); the proposed mapping until changed. */
export function SimpleTraining() {
  const state = useFetch<SimpleSettings>('/api/labels/simple')
  const [stain, setStain] = useState('')
  return (
    <section className="card simple-training" id="simple-training">
      <div className="row"><h2 style={{ margin: 0 }}>Simple labels in training</h2><span className="spacer" /><span className="badge">Fix the field (simple)</span></div>
      <p className="muted small">Labels from simple sessions are kept as people gave them. When a model trains on them, each simple class counts as the
        class chosen here, or not at all. Every model records the mapping it used.</p>
      <Loaded state={state} label="Loading the simple classes…">
        {(data) => {
          const stains = Object.keys(data.stains).filter((s) => data.stains[s].classes.length)
          const shown = stains.includes(stain) ? stain : stains[0]
          return <>
            <StainTabs stains={stains} shown={shown} onPick={setStain} />
            {shown && <MappingTable key={shown} stain={shown} data={data.stains[shown]} onSaved={state.reload} />}
          </>
        }}
      </Loaded>
    </section>
  )
}

function MappingTable({ stain, data, onSaved }: { stain: string; data: SimpleSettings['stains'][string]; onSaved: () => void }) {
  const [draft, setDraft] = useState<Record<string, string>>(data.training)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => setDraft(data.training), [data.training])
  const rows = [...data.classes.map((c) => ({ id: c.id, name: c.name, colour: c.colour })), { id: data.none.id, name: data.none.name, colour: '' }]
  const save = (training: Record<string, string>) => withError(async () => { await api.put(`/api/labels/simple/${stain}`, { training }); onSaved() }, setError)
  const changed = !same(draft, data.training)
  return (
    <>
      <table className="compact simple-mapping">
        <thead><tr><th>Simple class</th><th>Trains the {stain} model as</th><th className="muted">proposed</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.id}>
            <td>{r.colour ? <span className="look-swatch" style={{ borderColor: r.colour, background: r.colour }} /> : <span className="look-swatch look-removed" />} {r.name}</td>
            <td><select value={draft[r.id] ?? ''} onChange={(e) => setDraft({ ...draft, [r.id]: e.target.value })} aria-label={`${r.name} trains as`}>
              <option value="">not used</option>
              {data.targets.map((t) => <option key={t.value} value={t.value}>{t.name} ({t.role === 'positive' ? 'counted' : t.role === 'skip' ? 'left out' : 'not counted'})</option>)}
            </select></td>
            <td className="small muted">{data.targets.find((t) => t.value === data.proposed[r.id])?.name ?? 'not used'}</td>
          </tr>
        ))}</tbody>
      </table>
      <ErrorNote error={error} />
      <div className="row" style={{ marginTop: 8 }}>
        <button className="btn primary" disabled={!changed} onClick={() => void save(draft)}>Save the mapping</button>
        {changed && <button className="btn" onClick={() => setDraft(data.training)}>Undo changes</button>}
        <span className="spacer" />
        {!same(data.training, data.proposed) && <button className="link-btn small" onClick={() => void save(data.proposed)}>Back to the proposed mapping</button>}
      </div>
    </>
  )
}

/** "simple labels: plaque with cores → plaque · compact …": the mapping a model trained with, for its row on the Models page. */
export function mappingText(model: TrainedModel): string {
  const m = model.simple_mapping
  if (!m) return ''
  const pairs = Object.entries(m.training).filter(([id]) => (m.labels[id] ?? 0) > 0).map(([id, to]) => `${m.classes[id] ?? id} → ${to ? to.replace('/', ' · ') : 'not used'}`)
  return pairs.length ? `simple labels: ${pairs.join(', ')}` : ''
}

/** Settings: the simple classes of each stain (Advanced: change them). */
export function SimpleClassSettings() {
  const state = useFetch<SimpleSettings>('/api/labels/simple')
  const [stain, setStain] = useState('')
  return (
    <section className="card class-settings" id="simple-classes">
      <h2>Simple classes</h2>
      <p className="muted small">What people choose from in “Fix the field (simple)”: a few plain classes, worked one at a time. A candidate starts in
        a class when detection’s call is one it names. How they train is set on the Models page.</p>
      <Loaded state={state} label="Loading the simple classes…">
        {(data) => {
          const stains = Object.keys(data.stains)
          const shown = stains.includes(stain) ? stain : stains[0]
          return <>
            <StainTabs stains={stains} shown={shown} onPick={setStain} />
            {shown && <SimpleEditor key={shown} stain={shown} data={data.stains[shown]} onSaved={state.reload} />}
          </>
        }}
      </Loaded>
    </section>
  )
}

function SimpleEditor({ stain, data, onSaved }: { stain: string; data: SimpleSettings['stains'][string]; onSaved: () => void }) {
  const [draft, setDraft] = useState<SimpleClassDef[]>(data.classes)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => setDraft(data.classes), [data.classes])
  const change = (i: number, patch: Partial<SimpleClassDef>) => setDraft(draft.map((c, k) => (k === i ? { ...c, ...patch } : c)))
  const save = () => withError(async () => { await api.put(`/api/labels/simple/${stain}`, { classes: draft }); onSaved() }, setError)
  const reset = () => withError(async () => { await api.put(`/api/labels/simple/${stain}`, { reset: true }); onSaved() }, setError)
  const add = () => setDraft([...draft, { id: '', name: 'new class', description: '', key: String(draft.length + 1), colour: PALETTE.classes[0], kind: 'objects', starts_from: [], hidden: false }])
  return (
    <>
      <ul className="class-legend small simple-list">{data.classes.map((c) => (
        <li key={c.id}><span className="kbd key-slot">{c.key}</span><span className="look-swatch" style={{ borderColor: c.colour, background: c.colour }} />
          <b>{c.name}</b>{c.kind === 'wand' && <span className="badge">magic wand on threads</span>}
          <span className="muted">{c.starts_from.length ? `starts from ${c.starts_from.join(', ').replace(/\//g, ' · ')}` : 'starts empty'}</span></li>
      ))}</ul>
      <Advanced title="Change the simple classes" hint="names, descriptions, keys, colours, where candidates start, hide, add">
        <div className="simple-editor">{draft.map((c, i) => (
          <div key={i} className="simple-editor-row">
            <input type="text" value={c.name} aria-label="Name" onChange={(e) => change(i, { name: e.target.value })} />
            <input type="text" value={c.key} aria-label="Key" maxLength={1} style={{ width: 40 }} onChange={(e) => change(i, { key: e.target.value })} />
            <input type="color" value={c.colour} aria-label="Colour" onChange={(e) => change(i, { colour: e.target.value })} />
            <input type="text" value={c.description} aria-label="Description" placeholder="what it looks like" onChange={(e) => change(i, { description: e.target.value })} />
            {c.kind === 'objects' && <input type="text" value={c.starts_from.join(', ')} aria-label="Starts from" placeholder="starts from detection’s (e.g. plaque/compact)"
              onChange={(e) => change(i, { starts_from: e.target.value.split(',').map((t) => t.trim()).filter(Boolean) })} />}
            <label className="layer-toggle small"><input type="checkbox" checked={c.hidden} onChange={(e) => change(i, { hidden: e.target.checked })} />hidden</label>
            {!c.builtin && <button className="link-btn small" onClick={() => setDraft(draft.filter((_, k) => k !== i))}>remove</button>}
          </div>
        ))}</div>
        <ErrorNote error={error} />
        <div className="row" style={{ marginTop: 8 }}>
          <button className="btn" onClick={add}>Add a class</button>
          <button className="btn primary" disabled={same(draft, data.classes)} onClick={() => void save()}>Save</button>
          <span className="spacer" />
          <button className="link-btn small" onClick={() => void reset()}>Back to the built-in simple classes</button>
        </div>
      </Advanced>
    </>
  )
}
