import { useEffect, useState } from 'react'
import type { Group, ProjectConfig } from '../api'
import { api } from '../api'
import { ErrorNote, FilePicker, HelpBox, PageHead, PathLine, withError } from '../components'
import { useProject } from '../project'
import Welcome from './Welcome'

const SECTION_TEXT: Record<string, [string, string]> = {
  inputs: ['Study files', 'Tables stainID reads or makes during set-up. You rarely need to change these.'],
  models: ['Model files', 'Where each model is stored. The Models page can download or train them.'],
  outputs: ['Result folders', 'Where each step writes its results.'],
}

export default function Settings() {
  const { info, reload } = useProject()
  const [config, setConfig] = useState<ProjectConfig | null>(null)
  const [picking, setPicking] = useState<{ section: string; key: string } | null>(null)
  const [switching, setSwitching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  useEffect(() => { if (info) setConfig(structuredClone(info.config)) }, [info])
  if (!info || !config) return null

  const update = (next: ProjectConfig) => { setSaved(false); setConfig(next) }
  const setPath = (section: 'inputs' | 'models' | 'outputs', key: string, value: string) => update({ ...config, [section]: { ...config[section], [key]: value } })
  const setGroup = (i: number, group: Group) => update({ ...config, groups: config.groups.map((g, j) => (j === i ? group : g)) })
  const save = () => withError(async () => {
    const { name, pixel_size_um, field_context_px, tma, groups, inputs, models, outputs } = config
    await api.put('/api/project/config', { changes: { name, pixel_size_um, field_context_px, tma, groups, inputs, models, outputs } })
    setSaved(true)
    reload()
  }, setError)

  if (switching) return <><button className="btn" onClick={() => setSwitching(false)}>← Back to settings</button><Welcome /></>

  return (
    <>
      <PageHead title="Settings" subtitle={<>Saved in <code>{info.root}/stainid.yaml</code>. Paths are relative to the project folder.</>}>
        <button className="btn" onClick={() => setSwitching(true)}>Open or create another project…</button>
      </PageHead>
      <HelpBox id="settings">
        Most studies only need the first two boxes: the study name, how the TMA slides are laid out, and the diagnostic groups used in the TMA map.
        Everything else has sensible defaults inside the project folder.
      </HelpBox>

      <div className="grid grid-2">
        <div className="card">
          <h2>Study</h2>
          <label className="field">Study name<input type="text" value={config.name} onChange={(e) => update({ ...config, name: e.target.value })} /></label>
          <p className="muted small">{info.help.name}</p>
          <h2 style={{ marginTop: 16 }}>TMA layout</h2>
          <div className="row">
            <label className="field">Core name prefix<input type="text" value={config.tma.prefix} style={{ width: 90 }} onChange={(e) => update({ ...config, tma: { ...config.tma, prefix: e.target.value } })} /></label>
            <label className="field">Rows<input type="number" value={config.tma.rows} style={{ width: 70 }} onChange={(e) => update({ ...config, tma: { ...config.tma, rows: Number(e.target.value) } })} /></label>
            <label className="field">Columns<input type="number" value={config.tma.columns} style={{ width: 70 }} onChange={(e) => update({ ...config, tma: { ...config.tma, columns: Number(e.target.value) } })} /></label>
            <label className="field">Pixel size (µm)<input type="number" step="0.0001" value={config.pixel_size_um} style={{ width: 100 }} onChange={(e) => update({ ...config, pixel_size_um: Number(e.target.value) })} /></label>
          </div>
          <p className="muted small">Cores are named <code>{config.tma.prefix}3_B-2</code> (TMA 3, column B, row 2). {info.help['tma.rows']} {info.help['tma.columns']}
            {' '}Set these before <i>Find cores</i> and <i>Choose analysis fields</i>; changing them later needs those steps (and everything after) to be run again.</p>
        </div>

        <div className="card">
          <h2>Diagnostic groups</h2>
          <p className="muted small">{info.help.groups} Codes must match the <code>disease_group</code> column of your TMA map.</p>
          <table>
            <thead><tr><th>Code in TMA map</th><th>Shown as</th><th>Colour</th><th /></tr></thead>
            <tbody>{config.groups.map((g, i) => (
              <tr key={i}>
                <td><input type="text" value={g.code} style={{ width: 90 }} onChange={(e) => setGroup(i, { ...g, code: e.target.value })} /></td>
                <td><input type="text" value={g.label} style={{ width: 130 }} onChange={(e) => setGroup(i, { ...g, label: e.target.value })} /></td>
                <td><select value={g.color ?? i + 1} onChange={(e) => setGroup(i, { ...g, color: Number(e.target.value) })}>
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => <option key={n} value={n}>colour {n}</option>)}</select>
                  <span className="swatch" style={{ background: `var(--series-${g.color ?? i + 1})`, marginLeft: 6 }} /></td>
                <td className="row">
                  <button className="btn small" disabled={i === 0} onClick={() => { const g2 = [...config.groups]; [g2[i - 1], g2[i]] = [g2[i], g2[i - 1]]; update({ ...config, groups: g2 }) }}>↑</button>
                  <button className="btn small" onClick={() => update({ ...config, groups: config.groups.filter((_, j) => j !== i) })}>Remove</button>
                </td>
              </tr>
            ))}</tbody>
          </table>
          <button className="btn small" style={{ marginTop: 8 }} onClick={() => update({ ...config, groups: [...config.groups, { code: '', label: '' }] })}>Add group</button>
          {!config.groups.length && <p className="muted small">No groups set: codes from the TMA map are shown as they are.</p>}
        </div>
      </div>

      {(['inputs', 'models', 'outputs'] as const).map((section) => (
        <details key={section} className="card settings-section" open={section === 'inputs'}>
          <summary><h2 style={{ display: 'inline' }}>{SECTION_TEXT[section][0]}</h2> <span className="muted small">{SECTION_TEXT[section][1]}</span></summary>
          <table>
            <tbody>{Object.entries(config[section]).map(([key, value]) => (
              <tr key={key}>
                <td style={{ width: 220 }}><b>{key.replaceAll('_', ' ')}</b><div className="muted small wrap">{info.paths[section]?.[key]?.help}</div></td>
                <td><input type="text" value={value} style={{ width: '100%' }} onChange={(e) => setPath(section, key, e.target.value)} /></td>
                <td style={{ width: 90 }}><button className="btn small" onClick={() => setPicking({ section, key })}>Choose…</button></td>
                <td style={{ width: 260 }}>{info.paths[section]?.[key] && <PathLine path={info.paths[section][key].path} exists={info.paths[section][key].exists} />}</td>
              </tr>
            ))}</tbody>
          </table>
        </details>
      ))}

      <div className="save-bar">
        <button className="btn primary" onClick={save}>Save settings</button>
        {saved && <span className="status-finished small">Saved.</span>}
        <ErrorNote error={error} />
      </div>
      {picking && (
        <FilePicker title={`Choose ${picking.key.replaceAll('_', ' ')}`} pickFiles={/\.(csv|json|joblib|pth)$/.test(config[picking.section as 'inputs'][picking.key])}
          start={info.paths[picking.section]?.[picking.key]?.path ? `${info.root}/${info.paths[picking.section][picking.key].path}` : undefined}
          onClose={() => setPicking(null)} onPick={(path) => { setPath(picking.section as 'inputs', picking.key, path); setPicking(null) }} />
      )}
    </>
  )
}
