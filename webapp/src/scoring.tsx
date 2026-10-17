import { useState } from 'react'
import type { ImportedLabels, ScoreRow, Scores } from './api'
import { api, apiUrl } from './api'
import { ErrorNote, timeAgo, withError } from './components'
import { useFetch } from './hooks'
import { Advanced } from './setup/parts'

const percent = (value: number | null) => (value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`)
const two = (value: number | null) => (value === null || value === undefined ? '—' : value.toFixed(2))
const who = (rater: string) => (rater === 'consensus' ? 'Consensus of the raters' : rater)
const order = (a: ScoreRow, b: ScoreRow) => Number(b.rater === 'consensus') - Number(a.rater === 'consensus') || a.stain.localeCompare(b.stain) || a.rater.localeCompare(b.rater)

/** Models page: the external test. When the study was made from a dataset package with labels, the models in use are
scored against every rater and their consensus (what they found, how much of it was marked, F1, AUC), next to how
much the raters agree with each other. Shows nothing for a study without imported labels. */
export function ScoreOnLabels() {
  const state = useFetch<{ scores: Scores | null; has_labels: boolean }>('/api/models/score')
  const imported = useFetch<ImportedLabels>('/api/labels/imported')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  if (!state.data?.has_labels) return state.error ? <ErrorNote error={state.error} /> : null  // most studies have none: nothing to wait for
  const scores = state.data.scores
  const dataset = imported.data?.dataset
  const run = () => withError(async () => {
    setBusy(true)
    try { await api.post('/api/models/score', {}); state.reload() } finally { setBusy(false) }
  }, setError)
  const rows = [...(scores?.rows ?? [])].sort(order)
  return (
    <>
      <h2 className="stage-title">Score the models on these labels</h2>
      <div className="card score-card">
        <p className="secondary">{dataset ? <>The labels of <b>{dataset.title}</b>{dataset.licence ? ` (${dataset.licence})` : ''} </> : 'The imported labels '}
          were made by other people, in another lab. Comparing the models with them is a test on data the models never learned from.</p>
        {scores ? (
          <>
            <ul className="score-summary">{scores.summary.map((line) => <li key={line}>{line}</li>)}</ul>
            <div className="table-wrap"><table className="compact small score-table">
              <thead><tr><th>Labels of</th><th>Stain</th><th>What</th><th className="num">Marked</th><th className="num">Found</th><th className="num">Of detection's, marked</th>
                <th className="num">F1</th><th className="num">AUC</th></tr></thead>
              <tbody>{rows.map((r) => (
                <tr key={`${r.rater}-${r.stain}-${r.kind}-${r.class}`}>
                  <td>{who(r.rater)}</td><td>{r.stain}</td><td>{r.kind === 'image' ? `images with ${r.class}` : 'objects'}</td>
                  <td className="num">{r.marks.toLocaleString()}</td><td className="num" title={`${r.found} found, ${r.missed} missed`}>{percent(r.recall)}</td>
                  <td className="num" title={`${r.ai_objects} found by detection, ${r.ai_unmarked} of them not marked`}>{percent(r.precision)}</td>
                  <td className="num">{two(r.f1)}</td><td className="num">{two(r.auc)}</td>
                </tr>
              ))}</tbody>
            </table></div>
            <p className="small muted">“Found” (recall): the share of what people marked that stainID detection found. “Of detection's, marked” (precision): the share of what detection
              found that people marked. F1 combines both (1 is perfect). AUC: how well detection's confidence ranks marked objects above the rest (0.5 is chance).
              Scored {timeAgo(scores.time)}.</p>
          </>
        ) : <p className="small secondary">Not scored yet. Run the detection first (Set up → Run), then score.</p>}
        <div className="row">
          {scores && <a className="btn small" href={apiUrl('/api/models/score.csv')} download>Download the table (CSV)</a>}
          <span className="spacer" />
          <button className={`btn${scores ? '' : ' primary'}`} disabled={busy} onClick={run}>{busy ? 'Scoring…' : scores ? 'Score again' : 'Score the models'}</button>
        </div>
        <ErrorNote error={error} />
        {scores && (
          <Advanced title="How much the raters agree" hint="each pair of raters; matching rules">
            {scores.agreement.length ? (
              <div className="table-wrap"><table className="compact small">
                <thead><tr><th>Stain</th><th>Rater</th><th>and</th><th className="num">Images both labelled</th><th className="num">Objects: F1</th><th className="num">Images: kappa</th></tr></thead>
                <tbody>{scores.agreement.map((a) => (
                  <tr key={`${a.stain}-${a.rater}-${a.other}`}><td>{a.stain}</td><td>{a.rater}</td><td>{a.other}</td><td className="num">{a.shared_images.toLocaleString()}</td>
                    <td className="num">{two(a.object_f1)}</td><td className="num">{two(a.image_kappa)}</td></tr>
                ))}</tbody>
              </table></div>
            ) : <p className="small muted">One rater only: there is no agreement to measure.</p>}
            <p className="small muted">A mark is found when an object detection accepted lies within 12 µm (plaques), 8 µm (tau) or 6 µm (neurons) of it, or within the
              object's own size; a box when the object's centre is in it (the IoU ≥ 0.5 rule is in the CSV); an image label when detection accepted any object of
              that class in the image. Models scored: {Object.values(scores.models).join(', ')}.</p>
          </Advanced>
        )}
      </div>
    </>
  )
}
