import { useEffect, useState } from 'react'
import type { Group, ProjectConfig, ProjectInfo } from '../api'
import { api } from '../api'
import { CacheSpace } from '../cacheSpace'
import { ClassSettings } from '../labels/ClassSettings'
import { ErrorNote, FilePicker, HelpBox, Loading, PageHead, PathLine, withError } from '../components'
import type { Theme } from '../hooks'
import { useTheme } from '../hooks'
import { useProject } from '../project'
import { ApiTokens } from '../people'
import { UsersLink } from './Users'
import { ChoosePassword, ServerChoice, useSession } from '../session'
import { Advanced } from '../setup/parts'

type PathSection = 'inputs' | 'models' | 'outputs'
const PATH_SECTIONS: [PathSection, string, string][] = [
  ['inputs', 'Study files', 'Tables stainID reads or makes during set-up.'],
  ['models', 'Model files', 'Where each model is stored. The Models page can download or train them.'],
  ['outputs', 'Result folders', 'Where each step writes its results.'],
]
const COLOURS = [1, 2, 3, 4, 5, 6, 7, 8]
/** The settings this page edits (stainid.yaml may hold others; they are kept as they are). */
const EDITED = ['name', 'pixel_size_um', 'field_context_px', 'tma', 'groups', 'inputs', 'models', 'outputs'] as const

const edited = (config: ProjectConfig) => ({ ...Object.fromEntries(EDITED.map((key) => [key, config[key]])),
  labels: { apply_exclusions: !!(config as unknown as { labels?: { apply_exclusions?: boolean } }).labels?.apply_exclusions } })
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/** What is wrong with the settings, in plain words (empty when they can be saved). */
function problems(config: ProjectConfig): string[] {
  const out = []
  if (!config.name.trim()) out.push('The study needs a name.')
  const codes = config.groups.map((g) => g.code.trim())
  if (codes.some((c) => !c)) out.push('Every diagnostic group needs its code from the TMA map.')
  if (new Set(codes).size !== codes.length) out.push('Two diagnostic groups have the same code.')
  if (!(config.tma.rows >= 1 && config.tma.columns >= 1)) out.push('A TMA needs at least one row and one column.')
  if (!(config.pixel_size_um > 0)) out.push('The pixel size must be above 0.')
  return out
}

export default function Settings() {
  const { info } = useProject()
  return (
    <>
      <PageHead title="Settings" subtitle="How this study is set up, who can use stainID, and how the app looks.">
        <a className="btn" href="/projects">Open another project…</a>
      </PageHead>
      <HelpBox id="settings">
        Most studies only need the study name and the diagnostic groups; the TMA layout is usually filled in by the guided setup.
        Everything else has sensible defaults inside the project folder and is under Advanced. Nothing changes until you press <b>Save settings</b>.
      </HelpBox>
      {info ? <StudySettings info={info} /> : <Loading label="Loading the settings…" />}
      <ClassSettings />
      <UsersLink />
      <div className="grid grid-2 settings-more">
        <div id="password"><ChoosePassword title="Your password" current /></div>
        <Appearance />
      </div>
      <Advanced title="API tokens for scripts" hint="let a script or agent use this server as you">
        <ApiTokens />
      </Advanced>
      <Advanced title="Server and other computers" hint="which stainID server this page uses">
        <ServerSettings />
      </Advanced>
      <Advanced title="Space for pictures the app makes" hint="how much disk space they may take; clear them">
        <CacheSpace />
      </Advanced>
    </>
  )
}

function StudySettings({ info }: { info: ProjectInfo }) {
  const { reload } = useProject()
  const [config, setConfig] = useState<ProjectConfig>(() => structuredClone(info.config))
  const [picking, setPicking] = useState<{ section: PathSection; key: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [state, setState] = useState<'idle' | 'saving' | 'saved'>('idle')
  const [base, setBase] = useState(info.config)
  useEffect(() => {  // the settings were saved (here, or by someone else): take them, unless this page has unsaved edits
    if (info.config === base) return
    setConfig((current) => (same(edited(current), edited(base)) ? structuredClone(info.config) : current))
    setBase(info.config)
  }, [info.config, base])
  const dirty = !same(edited(config), edited(info.config))
  useEffect(() => {
    if (!dirty) return
    const warn = (e: BeforeUnloadEvent) => e.preventDefault()
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  const update = (next: ProjectConfig) => { setState('idle'); setConfig(next) }
  const setPath = (section: PathSection, key: string, value: string) => update({ ...config, [section]: { ...config[section], [key]: value } })
  const issues = problems(config)
  const redo = [
    !same(config.tma, info.config.tma) && 'the TMA layout',
  ].filter(Boolean)
  const save = async () => {
    setState('saving')
    const done = await withError(() => api.put('/api/project/config', { changes: edited(config) }), setError)
    setState(done ? 'saved' : 'idle')
    if (done) reload()
  }

  return (
    <>
      <div className="grid grid-2">
        <div className="card settings-card">
          <h2>Study</h2>
          <label className="field">Study name<input type="text" value={config.name} onChange={(e) => update({ ...config, name: e.target.value })} /></label>
          <p className="muted small">{info.help.name}</p>
          <GroupsEditor groups={config.groups} help={info.help.groups} onChange={(groups) => update({ ...config, groups })} />
        </div>
        <TmaSettings config={config} info={info} onChange={update} />
      </div>

      <Advanced title="More study settings" hint="scan pixel size, tile margin, file locations">
        <div className="row">
          <label className="field">Pixel size (µm per pixel)<input type="number" step="0.0001" min="0" value={config.pixel_size_um} style={{ width: 120 }}
            onChange={(e) => update({ ...config, pixel_size_um: Number(e.target.value) })} /></label>
          <label className="field">Tile margin (pixels)<input type="number" min="0" value={config.field_context_px} style={{ width: 120 }}
            onChange={(e) => update({ ...config, field_context_px: Number(e.target.value) })} /></label>
        </div>
        <p className="muted small" style={{ margin: 0 }}>{info.help.pixel_size_um} {info.help.field_context_px}</p>
        {PATH_SECTIONS.map(([section, title, text]) => (
          <section key={section} className="settings-paths">
            <h3>{title}</h3>
            <p className="muted small">{text} Paths are relative to the project folder, <code>{info.root}</code>.</p>
            <div className="table-wrap" style={{ maxHeight: 'none' }}>
              <table>
                <tbody>{Object.entries(config[section]).filter(([key]) => !info.paths[section] || key in info.paths[section]).map(([key, value]) => {
                  const where = info.paths[section]?.[key]
                  return (
                    <tr key={key}>
                      <td className="wrap" style={{ width: '28%' }}><b>{key.replaceAll('_', ' ')}</b>{where?.help && <div className="muted small">{where.help}</div>}</td>
                      <td><input type="text" value={value} style={{ width: '100%', minWidth: 160 }} aria-label={key.replaceAll('_', ' ')} onChange={(e) => setPath(section, key, e.target.value)} /></td>
                      <td style={{ width: 90 }}><button className="btn small" onClick={() => setPicking({ section, key })}>Choose…</button></td>
                      <td className="settings-exists">{where && <PathLine path={where.path} exists={where.exists} />}</td>
                    </tr>
                  )
                })}</tbody>
              </table>
            </div>
          </section>
        ))}
      </Advanced>

      <div className={`save-bar${dirty ? ' dirty' : ''}`} role="status" aria-live="polite">
        <button className="btn primary" onClick={save} disabled={!dirty || issues.length > 0 || state === 'saving'}>{state === 'saving' ? 'Saving…' : 'Save settings'}</button>
        {dirty && <button className="btn" onClick={() => update(structuredClone(info.config))} disabled={state === 'saving'}>Undo changes</button>}
        <span className="small">
          {issues.length ? <span className="level-warning">{issues.join(' ')}</span>
            : dirty ? <span className="secondary">You have unsaved changes.{redo.length ? ` Changing ${redo.join(' and ')} means the steps that use it must run again (Run page, Start over).` : ''}</span>
              : state === 'saved' ? <span className="status-finished">✓ Saved to stainid.yaml.</span>
                : <span className="muted">No unsaved changes.</span>}
        </span>
      </div>
      <ErrorNote error={error} />
      {picking && (
        <FilePicker title={`Choose ${picking.key.replaceAll('_', ' ')}`} pickFiles={/\.(csv|json|joblib|pth)$/.test(config[picking.section][picking.key])}
          start={info.paths[picking.section]?.[picking.key]?.path ? `${info.root}/${info.paths[picking.section][picking.key].path}` : undefined}
          onClose={() => setPicking(null)} onPick={(path) => { setPath(picking.section, picking.key, path); setPicking(null) }} />
      )}
    </>
  )
}

/** The diagnostic groups: the code used in the TMA map, the name shown, the colour in plots, and their order. */
function GroupsEditor({ groups, help, onChange }: { groups: Group[]; help: string; onChange: (groups: Group[]) => void }) {
  const set = (i: number, group: Group) => onChange(groups.map((g, j) => (j === i ? group : g)))
  const move = (i: number, by: number) => { const next = [...groups]; [next[i], next[i + by]] = [next[i + by], next[i]]; onChange(next) }
  return (
    <>
      <h2 style={{ marginTop: 18 }}>Diagnostic groups</h2>
      <p className="muted small">{help} Codes must match the <code>disease_group</code> column of the TMA map. Plots use this order and these colours.</p>
      {groups.length > 0 && (
        <table className="groups-table">
          <thead><tr><th>Code in TMA map</th><th>Shown as</th><th>Colour</th><th><span className="sr-only">Order</span></th></tr></thead>
          <tbody>{groups.map((g, i) => {
            const colour = g.color ?? i + 1
            return (
              <tr key={i}>
                <td><input type="text" value={g.code} style={{ width: 90 }} aria-label="Code in TMA map" onChange={(e) => set(i, { ...g, code: e.target.value })} /></td>
                <td><input type="text" value={g.label} style={{ width: 130 }} aria-label="Shown as" onChange={(e) => set(i, { ...g, label: e.target.value })} /></td>
                <td>
                  <div className="colour-choice" role="radiogroup" aria-label={`Colour of ${g.label || g.code || 'this group'}`}>
                    {COLOURS.map((n) => (
                      <button key={n} role="radio" aria-checked={colour === n} aria-label={`colour ${n}`} className={colour === n ? 'chosen' : ''}
                        style={{ background: `var(--series-${n})` }} onClick={() => set(i, { ...g, color: n })} />
                    ))}
                  </div>
                </td>
                <td><div className="row" style={{ flexWrap: 'nowrap', gap: 6 }}>
                  <button className="btn small" disabled={i === 0} aria-label="Move up" title="Move up" onClick={() => move(i, -1)}>↑</button>
                  <button className="btn small" disabled={i === groups.length - 1} aria-label="Move down" title="Move down" onClick={() => move(i, 1)}>↓</button>
                  <button className="link-btn small" onClick={() => onChange(groups.filter((_, j) => j !== i))}>Remove</button>
                </div></td>
              </tr>
            )
          })}</tbody>
        </table>
      )}
      {!groups.length && <p className="muted small">No groups yet: the codes in the TMA map are shown as they are.</p>}
      <button className="btn small" style={{ marginTop: 8 }} onClick={() => onChange([...groups, { code: '', label: '' }])}>Add a group</button>
    </>
  )
}

function TmaSettings({ config, info, onChange }: { config: ProjectConfig; info: ProjectInfo; onChange: (config: ProjectConfig) => void }) {
  const tma = (change: Partial<ProjectConfig['tma']>) => onChange({ ...config, tma: { ...config.tma, ...change } })
  return (
    <div className="card settings-card">
      <h2>TMA slides</h2>
      <div className="row">
        <label className="field">Core name prefix<input type="text" value={config.tma.prefix} style={{ width: 90 }} onChange={(e) => tma({ prefix: e.target.value })} /></label>
        <label className="field">Rows<input type="number" min="1" value={config.tma.rows} style={{ width: 70 }} onChange={(e) => tma({ rows: Number(e.target.value) })} /></label>
        <label className="field">Columns<input type="number" min="1" value={config.tma.columns} style={{ width: 70 }} onChange={(e) => tma({ columns: Number(e.target.value) })} /></label>
      </div>
      <p className="muted small">Cores are named like <code>{config.tma.prefix}3_B-2</code>: TMA 3, column B, row 2. {info.help['tma.rows']} {info.help['tma.columns']}
        {' '}Set these before finding the cores; changing them later means finding the cores and the steps after it must run again.</p>
    </div>
  )
}

function Appearance() {
  const [theme, setTheme] = useTheme()
  const options: [Theme, string][] = [['system', 'Same as this computer'], ['light', 'Light'], ['dark', 'Dark']]
  return (
    <div className="card">
      <h2>Appearance</h2>
      <p className="muted small">For this browser only; other people keep their own choice.</p>
      <fieldset className="radio-list">
        <legend className="sr-only">Colour theme</legend>
        {options.map(([value, text]) => (
          <label key={value}><input type="radio" name="theme" checked={theme === value} onChange={() => setTheme(value)} />{text}</label>
        ))}
      </fieldset>
    </div>
  )
}

function ServerSettings() {
  const { session } = useSession()
  return (
    <>
      <ServerChoice />
      <p className="muted small" style={{ margin: 0 }}>People on your network can simply open this server's address in their browser: each signs in, and several
        can work at the same time. Web apps served from other addresses that this server answers:{' '}
        {session.allowed_origins?.length ? session.allowed_origins.map((o) => <code key={o}>{o} </code>) : 'none'}
        {' '}(set with <code>stainid serve --allow-origin URL</code> or <code>STAINID_ALLOW_ORIGINS</code>).</p>
    </>
  )
}
