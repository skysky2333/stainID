/* Making an annotation round (the Labels page): stain, round, where its objects come from (every ready core, some
   TMAs, or fields chosen on the cores, each core showing whether it is ready) and how many. Everything else (how
   generous the candidates are, how many people must decide each object, the session ID, a note, types) is under
   Advanced, with the older kinds of session (whole-field review and quick check of the label pool, outline every object). */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { RoundScope, SessionInfo, Vocabulary } from '../api'
import { api } from '../api'
import { ErrorNote, Loaded, Modal, withError } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import { FieldPicker } from './fields'
import { ROUND_TEXT, ROUND_TITLE } from './rounds'

type Where = 'all' | 'tmas' | 'pick'
/** What the project's pictures and their groups are called: cores on TMAs, or slides / images in batches. */
const WORDS = { tma: { pictures: 'cores', groups: 'TMAs', group: 'TMA' }, slides: { pictures: 'slides', groups: 'batches', group: 'batch' },
  images: { pictures: 'images', groups: 'batches', group: 'batch' } } as const
const LEVEL_TEXT: Record<string, string> = {
  all: 'Every candidate the AI proposed (most generous)', '0.5': 'Proposed at half the stain threshold or stronger',
  '0.75': 'At three quarters of the stain threshold or stronger', '1': 'At the stain threshold or stronger (as the AI counts)',
  '1.5': 'Only clearly stained ones (1.5 × the threshold)',
}

export function NewRoundSession({ stains, onClose, onCreated, onOlder }: {
  stains: string[]; onClose: () => void; onCreated: (s: SessionInfo) => void; onOlder: () => void
}) {
  const { tmaName, info } = useProject()
  const words = WORDS[info?.kind ?? 'tma'] ?? WORDS.tma
  const [stain, setStain] = useState(stains[0] ?? 'NeuN')
  const [round, setRound] = useState<1 | 2>(1)
  const [where, setWhere] = useState<Where>('all')
  const [tmas, setTmas] = useState<string[]>([])
  const [chosen, setChosen] = useState<string[]>([])
  const [objects, setObjects] = useState(500)
  const [fieldCount, setFieldCount] = useState(10)
  const [title, setTitle] = useState('')
  const [open, setOpen] = useState(true)
  const [level, setLevel] = useState('all')
  const [needed, setNeeded] = useState(1)
  const [name, setName] = useState('')
  const [instructions, setInstructions] = useState('')
  const [subtypes, setSubtypes] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const scope = useFetch<RoundScope>(`/api/labels/rounds/scope?stain=${encodeURIComponent(stain)}`)
  const defaults = useFetch<{ vocabularies: Record<string, Vocabulary> }>('/api/labels/defaults')
  const placeholder = `${stain} · ${ROUND_TITLE[round]} – ${new Date().toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}`
  const nothing = (where === 'pick' && !chosen.length) || (where === 'tmas' && !tmas.length)
  const create = async () => {
    setBusy(true)
    const body = { stain, round, title: title.trim() || placeholder, name, open, instructions, subtypes, labels_needed: needed,
      level: level === 'all' ? null : Number(level), n_objects: objects, n_fields: fieldCount,
      tmas: where === 'tmas' ? tmas : null, tile_ids: where === 'pick' ? chosen : null }
    const made = await withError(() => api.post<SessionInfo>('/api/labels/sessions', body), setError)
    setBusy(false)
    if (made) { onCreated(made); onClose() }
  }
  const vocabulary = defaults.data?.vocabularies[stain]
  const typed = vocabulary?.classes.filter((c) => !c.hidden && c.subtypes.some((s) => !s.hidden)) ?? []
  return (
    <Modal title="New labelling session" onClose={onClose} wide>
      <div className="new-session">
        <fieldset className="field"><legend>Stain</legend>
          <div className="segmented">{stains.map((s) => (
            <button key={s} type="button" className={`btn${stain === s ? ' chosen' : ''}`} aria-pressed={stain === s}
              onClick={() => { setStain(s); setChosen([]); setTmas([]) }}>{s}</button>
          ))}</div>
        </fieldset>
        <fieldset className="field"><legend>What should people do?</legend>
          <div className="choice-grid">
            {([1, 2] as const).map((r) => (
              <label key={r} className={`choice-card${round === r ? ' chosen' : ''}`}>
                <input type="radio" name="round" checked={round === r} onChange={() => setRound(r)} />
                <span><b>Round {r} · {ROUND_TITLE[r]}{r === 1 ? ' (start here)' : ''}</b><span className="small secondary">{ROUND_TEXT[r]}</span></span>
              </label>
            ))}
          </div>
          <p className="small muted" style={{ margin: '6px 0 0' }}>People work one field at a time: a field someone is on, or has finished, is not handed to anyone else. Both rounds share one set of answers.</p>
        </fieldset>
        <Loaded state={scope} label="Looking at which cores are ready…">
          {(s) => <>
            <fieldset className="field"><legend>Which {words.pictures}?</legend>
              <p className={`small ${s.cores_ready ? 'secondary' : 'warning-text'}`} style={{ margin: '0 0 8px' }}>
                {s.cores_ready ? <>{s.cores_ready.toLocaleString()} of {s.cores.toLocaleString()} {words.pictures} have the AI’s candidates ({s.fields_ready.toLocaleString()} fields).</>
                  : <>No {stain} {words.pictures.slice(0, -1)} has the AI’s candidates yet: run the v2 detection first. Until then the older kinds of session (under Advanced) use the current label pool.</>}
                {s.from_pool && <span className="muted"> These candidates were made from the label pool for trying the rounds out.</span>}</p>
              <div className="segmented">
                <button type="button" className={`btn${where === 'all' ? ' chosen' : ''}`} onClick={() => setWhere('all')}>Every ready {words.pictures.slice(0, -1)}</button>
                <button type="button" className={`btn${where === 'tmas' ? ' chosen' : ''}`} onClick={() => setWhere('tmas')}>Some {words.groups}</button>
                <button type="button" className={`btn${where === 'pick' ? ' chosen' : ''}`} onClick={() => setWhere('pick')}>Choose on the {words.pictures}</button>
              </div>
              {where === 'tmas' && <div className="tma-chips">{s.tmas.map((t) => (
                <label key={t.tma} className={`tma-chip${t.cores_ready ? '' : ' is-blocked'}`} title={`${t.fields_ready} of ${t.fields} fields ready`}>
                  <input type="checkbox" disabled={!t.cores_ready} checked={tmas.includes(t.tma)}
                    onChange={() => setTmas(tmas.includes(t.tma) ? tmas.filter((x) => x !== t.tma) : [...tmas, t.tma])} />
                  <b>{info?.kind === 'tma' || !info?.kind ? tmaName(t.tma) : t.tma}</b> <span className="muted small">{t.cores_ready} of {t.cores} {words.pictures} ready</span>
                </label>
              ))}</div>}
              {where === 'pick' && <FieldPicker stain={stain} chosen={chosen} onChange={setChosen} source="candidates" />}
            </fieldset>
            {where !== 'pick' || round === 1 ? (
              <label className="field inline-number">{round === 1 ? 'How many objects (0 = every one)' : 'How many fields'}
                {round === 1
                  ? <input type="number" min={0} value={objects} style={{ width: 100 }} onChange={(e) => setObjects(Math.max(0, Number(e.target.value)))} />
                  : <input type="number" min={1} max={500} value={fieldCount} style={{ width: 100 }} onChange={(e) => setFieldCount(Math.max(1, Number(e.target.value)))} />}
                <span className="small muted">{round === 1 ? 'The most informative first: where the AI is unsure, then strong ones it rejected, then weak ones it accepted.'
                  : 'Fields nobody has checked, spread over TMAs and groups.'}</span>
              </label>
            ) : null}
          </>}
        </Loaded>
        <label className="field">Title people see<input type="text" value={title} placeholder={placeholder} onChange={(e) => setTitle(e.target.value)} /></label>
        <label className="layer-toggle"><input type="checkbox" checked={open} onChange={(e) => setOpen(e.target.checked)} />Open to labellers right away</label>
        <Advanced hint="how generous, people per field, session ID, note, types, older kinds of session">
          <label className="field">How generous (which candidates are shown)
            <select value={level} onChange={(e) => setLevel(e.target.value)}>
              {Object.keys(scope.data?.levels ?? LEVEL_TEXT).map((k) => <option key={k} value={k}>{LEVEL_TEXT[k] ?? k}</option>)}</select></label>
          <div className="row" style={{ alignItems: 'flex-end' }}>
            <label className="field">People who must finish each field<input type="number" min={1} max={9} value={needed} style={{ width: 80 }}
              onChange={(e) => setNeeded(Math.max(1, Number(e.target.value)))} /></label>
            <label className="field">Session ID (in links and files)<input type="text" value={name} placeholder="made from the title"
              onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
          </div>
          <label className="field">Note for labellers (optional)<textarea rows={2} value={instructions} placeholder="e.g. Count only neurons with a visible nucleus."
            onChange={(e) => setInstructions(e.target.value)} /></label>
          {vocabulary && <p className="muted small" style={{ margin: 0 }}>Classes: {vocabulary.classes.filter((c) => !c.hidden).map((c) => c.name).join(' · ')}.
            {' '}<Link to="/settings#label-classes">Change the classes</Link></p>}
          {typed.length > 0 && <label className="layer-toggle"><input type="checkbox" checked={subtypes} onChange={(e) => setSubtypes(e.target.checked)} />
            Also ask for the type ({typed.map((c) => c.name).join(', ')})</label>}
          <p className="small secondary" style={{ marginBottom: 0 }}>Older kinds of session — the whole-field review and quick check on the current label pool, and
            “Outline every object” (a blind segmentation check): <button className="link-btn" onClick={onOlder}>make one of those instead…</button></p>
        </Advanced>
        <ErrorNote error={error ?? scope.error} />
        <div className="row"><span className="spacer" />
          {nothing && <span className="small muted">{where === 'pick' ? `Click squares on the ${words.pictures} above to choose fields.` : `Tick at least one ${words.group}.`}</span>}
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || nothing || !scope.data?.cores_ready} onClick={create}>{busy ? 'Creating…' : 'Create session'}</button>
        </div>
      </div>
    </Modal>
  )
}
