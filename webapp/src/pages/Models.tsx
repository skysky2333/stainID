import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { Metrics, ModelInfo, ReviewSet, Step, TrainedModel } from '../api'
import { api, runStep } from '../api'
import { ErrorNote, HelpBox, PageHead, PathLine, Progress, Term, fmt, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { StepCard } from './Workflow'

const TITLES: Record<string, string> = {
  neun: 'NeuN model', amyloid: '6E10 model', tau: 'AT8 model', cellpose: 'Cellpose-SAM', sam: 'Segment Anything',
  huggingface_home: 'Phikon', plaque_cnn_dir: 'Published plaque CNNs',
}
const DETAILS: Record<string, string> = {
  n_features: 'measurements per object', training_labels: 'training labels', training_n: 'training labels', training_positive_n: 'positive labels',
  probability_threshold: 'counts an object at probability ≥', context_threshold: 'counts an object at probability ≥', identity_threshold: 'counts a plaque at probability ≥',
  morphotype_threshold: 'compact if core stain ≥ × slide threshold', morphotype_minimum_diameter_um: 'compact / diffuse from diameter (µm)',
  nms_radius_um: 'merges detections closer than (µm)',
}

export default function Models() {
  const { reload: reloadProject } = useProject()
  const models = useFetch<ModelInfo[]>('/api/models')
  const steps = useFetch<Step[]>('/api/steps', 3000)
  const trained = useFetch<TrainedModel[]>('/api/models/trained', 5000)
  const sets = useFetch<ReviewSet[]>('/api/reviews', 5000)
  const [error, setError] = useState<string | null>(null)
  const step = (id: string) => steps.data?.find((s) => s.id === id)
  const training = (sets.data ?? []).filter((s) => s.purpose === 'training')

  const download = (key: string) => withError(async () => { await runStep('download_models', { which: [key] }); steps.reload() }, setError)
  const activate = (path: string) => withError(async () => {
    await api.post('/api/models/activate', { path })
    trained.reload(); models.reload(); reloadProject()
  }, setError)

  return (
    <>
      <PageHead title="Models" subtitle="The models that decide what counts as a neuron, a plaque or a tau+ neuron — and how to train your own." />
      <HelpBox id="models">
        <p>stainID uses two kinds of models:</p>
        <ul>
          <li><b>Published models</b> (Cellpose-SAM, Segment Anything, Phikon) are the same for every study. Press <b>Download</b> if one is missing.</li>
          <li><b>Stain models</b> (NeuN, 6E10, AT8) learn what a real neuron / plaque / tau+ neuron looks like with <i>your</i> staining.
            They are <Term term="random forest">random forests</Term> trained on objects you label. To train or improve one:
            <b> 1</b> create a <Term>training set</Term>, <b>2</b> label it, <b>3</b> train, then compare it with the current model and choose which to use.</li>
        </ul>
        <p className="muted small">Switching to a new model does not change results you already have: run the detection step again with “Start over” to redo them.</p>
      </HelpBox>
      <ErrorNote error={error ?? models.error} />

      <h2 className="stage-title">Models in use</h2>
      <div className="grid grid-3">
        {(models.data ?? []).map((m) => (
          <div key={m.key} className="card model-card">
            <div className="row"><h3 className="step-title">{TITLES[m.key] ?? m.key}</h3><span className="spacer" />
              <span className={`badge ${m.exists ? 'status-finished' : 'status-failed'}`}>{m.exists ? 'ready' : 'missing'}</span></div>
            <p className="secondary small">{m.role}</p>
            <PathLine path={m.path} exists={m.exists} />
            {m.bundle && (
              <table className="compact"><tbody>
                {Object.entries(m.bundle).filter(([k]) => k in DETAILS).map(([k, v]) => (
                  <tr key={k}><td className="secondary">{DETAILS[k]}</td><td className="num">{typeof v === 'number' ? +v.toPrecision(3) : String(v)}</td></tr>
                ))}
              </tbody></table>
            )}
            {!m.exists && m.source === 'download' && <button className="btn primary small" onClick={() => download(m.key)}>Download</button>}
            {!m.exists && m.source === 'train' && <a className="btn small" href="#train">Train one below</a>}
            {!m.exists && m.source === 'manual' && <p className="muted small">Optional. Copy the published weights (Wong et al. 2022; Tang et al. 2019) into this folder.</p>}
          </div>
        ))}
      </div>
      {step('download_models')?.job && <div style={{ marginTop: 12 }}><StepCard step={step('download_models')!} steps={steps.data ?? []} onChanged={steps.reload} /></div>}

      <h2 className="stage-title" id="train">Train a stain model</h2>
      <div className="train-flow">
        <div>
          <div className="flow-number">1</div>
          {step('training_set') && <StepCard step={step('training_set')!} steps={steps.data ?? []} onChanged={() => { steps.reload(); sets.reload() }} />}
        </div>
        <div>
          <div className="flow-number">2</div>
          <div className="card">
            <h3 className="step-title">Label the training sets</h3>
            <p className="step-summary">Open a set and say what each object is. Diagnosis and the model’s answer are hidden while you label.</p>
            {training.length ? (
              <table>
                <thead><tr><th>Set</th><th>Stain</th><th style={{ width: 200 }}>Labelled</th><th /></tr></thead>
                <tbody>{training.map((s) => (
                  <tr key={s.name}>
                    <td>{s.title}<div className="muted small">{Object.entries(s.label_counts).map(([k, v]) => `${k}: ${v}`).join(' · ')}</div></td>
                    <td>{s.stain}</td>
                    <td><Progress done={s.labelled} total={s.items} /><span className="muted small">{s.labelled} of {s.items}</span></td>
                    <td><Link className="btn small primary" to={`/label/${s.name}`}>Label</Link></td>
                  </tr>
                ))}</tbody>
              </table>
            ) : <p className="muted small">No training sets yet — create one in step 1.</p>}
          </div>
        </div>
        <div>
          <div className="flow-number">3</div>
          {step('train') && <StepCard step={step('train')!} steps={steps.data ?? []} onChanged={() => { steps.reload(); trained.reload() }} />}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>Trained models</h2>
        <p className="muted small">
          Each new model is tested with <Term>leave-one-TMA-out</Term> checks on your labels; the current model is scored on the same labels.
          <Term> precision</Term>: of the objects counted, how many are right. <Term>recall</Term>: of the real objects, how many are found.
          <Term> AUC</Term>: overall ranking quality (1 = perfect). Prefer the model with higher precision <i>and</i> recall.
        </p>
        {trained.data && trained.data.length ? (
          <table>
            <thead><tr><th>Stain</th><th>Trained</th><th>Labels</th><th>New model</th><th>Current model on the same labels</th><th /></tr></thead>
            <tbody>{trained.data.map((t) => (
              <tr key={t.path}>
                <td><b>{t.stain}</b></td>
                <td className="small">{timeAgo(t.created)}<div className="muted">{t.sets.join(', ')}</div></td>
                <td className="num">{t.new_model.n}<div className="muted small">{t.new_model.positives} positive</div></td>
                <td><MetricLine m={t.new_model} /><div className="muted small">{t.validation}</div></td>
                <td>{t.current_model ? <MetricLine m={t.current_model} /> : <span className="muted small">no current model</span>}</td>
                <td>{t.active ? <span className="badge status-finished">in use</span> : <button className="btn small primary" onClick={() => activate(t.path)}>Use this model</button>}</td>
              </tr>
            ))}</tbody>
          </table>
        ) : <p className="muted small">No trained models yet.</p>}
      </div>
    </>
  )
}

function MetricLine({ m }: { m: Metrics }) {
  return <span className="small">AUC <b>{fmt(m.auc, 2)}</b> · precision <b>{fmt(m.precision, 2)}</b> · recall <b>{fmt(m.recall, 2)}</b></span>
}
