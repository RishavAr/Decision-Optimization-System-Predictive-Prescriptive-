import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell, ScatterChart, Scatter, Area, AreaChart, ComposedChart, ReferenceLine, PieChart, Pie } from 'recharts';

// Data from the optimization system
const thresholdData = [
  { threshold: 0.05, profit: -15000, approvalRate: 5, defaultRate: 0 },
  { threshold: 0.10, profit: 8000, approvalRate: 12, defaultRate: 3.3 },
  { threshold: 0.15, profit: 17972, approvalRate: 18, defaultRate: 6.7 },
  { threshold: 0.168, profit: 24774, approvalRate: 24.8, defaultRate: 6.5 },
  { threshold: 0.20, profit: 15000, approvalRate: 32, defaultRate: 9.4 },
  { threshold: 0.25, profit: -18842, approvalRate: 49.2, defaultRate: 15.4 },
  { threshold: 0.30, profit: -35000, approvalRate: 58, defaultRate: 17.2 },
  { threshold: 0.35, profit: -55650, approvalRate: 69.2, defaultRate: 19.7 },
  { threshold: 0.40, profit: -75000, approvalRate: 78, defaultRate: 21.5 },
  { threshold: 0.50, profit: -132222, approvalRate: 96, defaultRate: 24.6 },
  { threshold: 0.70, profit: -140118, approvalRate: 100, defaultRate: 25.6 },
];

const policyData = [
  { policy: 'Ultra Conservative', threshold: 0.15, profit: 17972, approvalRate: 18, defaultRate: 6.7, color: '#10b981' },
  { policy: 'Conservative', threshold: 0.25, profit: -18842, approvalRate: 49.2, defaultRate: 15.4, color: '#22c55e' },
  { policy: 'Moderate', threshold: 0.35, profit: -55650, approvalRate: 69.2, defaultRate: 19.7, color: '#f59e0b' },
  { policy: 'Aggressive', threshold: 0.50, profit: -132222, approvalRate: 96, defaultRate: 24.6, color: '#ef4444' },
  { policy: 'Ultra Aggressive', threshold: 0.70, profit: -140118, approvalRate: 100, defaultRate: 25.6, color: '#dc2626' },
  { policy: 'Optimal', threshold: 0.168, profit: 24774, approvalRate: 24.8, defaultRate: 6.5, color: '#3b82f6' },
];

const stressData = [
  { scenario: 'Baseline', multiplier: 1.0, profit: 24774, defaultRate: 6.5 },
  { scenario: 'Mild Recession', multiplier: 1.25, profit: 9559, defaultRate: 7.1 },
  { scenario: 'Moderate', multiplier: 1.5, profit: 3519, defaultRate: 7.1 },
  { scenario: 'Severe', multiplier: 2.0, profit: 2838, defaultRate: 8.5 },
  { scenario: 'Crisis', multiplier: 3.0, profit: 256, defaultRate: 12.0 },
];

const modelData = [
  { model: 'Logistic Regression', auc: 0.6425, cv: 0.5472 },
  { model: 'Random Forest', auc: 0.6696, cv: 0.6300 },
  { model: 'Gradient Boosting', auc: 0.6172, cv: 0.6178 },
];

const featureImportance = [
  { feature: 'checking_balance', importance: 0.18 },
  { feature: 'credit_history', importance: 0.15 },
  { feature: 'credit_amount', importance: 0.12 },
  { feature: 'duration_months', importance: 0.11 },
  { feature: 'employment', importance: 0.09 },
  { feature: 'age', importance: 0.08 },
  { feature: 'savings_balance', importance: 0.07 },
  { feature: 'purpose', importance: 0.06 },
  { feature: 'housing', importance: 0.05 },
  { feature: 'installment_rate', importance: 0.04 },
];

const interestSensitivity = [
  { rate: 4, profit: -5000 },
  { rate: 5, profit: 5000 },
  { rate: 6, profit: 12000 },
  { rate: 7, profit: 18000 },
  { rate: 8, profit: 24774 },
  { rate: 9, profit: 32000 },
  { rate: 10, profit: 40000 },
  { rate: 12, profit: 55000 },
  { rate: 15, profit: 78000 },
];

const formatCurrency = (value) => {
  if (value >= 0) return `$${(value/1000).toFixed(0)}K`;
  return `-$${Math.abs(value/1000).toFixed(0)}K`;
};

const formatPercent = (value) => `${value.toFixed(1)}%`;

export default function DecisionOptimizationDashboard() {
  const [selectedPolicy, setSelectedPolicy] = useState('Optimal');
  const [activeTab, setActiveTab] = useState('overview');

  const selectedPolicyData = policyData.find(p => p.policy === selectedPolicy);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'threshold', label: 'Threshold Optimization', icon: '🎯' },
    { id: 'policies', label: 'Policy Comparison', icon: '⚖️' },
    { id: 'stress', label: 'Stress Testing', icon: '🔬' },
    { id: 'features', label: 'Risk Drivers', icon: '📈' },
  ];

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
      fontFamily: "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
      color: '#e2e8f0',
      padding: '24px',
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
        border: '1px solid rgba(59, 130, 246, 0.3)',
        borderRadius: '16px',
        padding: '24px 32px',
        marginBottom: '24px',
        backdropFilter: 'blur(10px)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h1 style={{ 
              fontSize: '28px', 
              fontWeight: '700', 
              margin: 0,
              background: 'linear-gradient(135deg, #60a5fa, #a78bfa)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.5px'
            }}>
              Decision Optimization System
            </h1>
            <p style={{ margin: '8px 0 0', color: '#94a3b8', fontSize: '14px' }}>
              Predictive + Prescriptive Analytics for Loan Approval Strategy
            </p>
          </div>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{
              background: 'rgba(16, 185, 129, 0.15)',
              border: '1px solid rgba(16, 185, 129, 0.4)',
              borderRadius: '12px',
              padding: '12px 20px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '11px', color: '#6ee7b7', textTransform: 'uppercase', letterSpacing: '1px' }}>Optimal Profit</div>
              <div style={{ fontSize: '24px', fontWeight: '700', color: '#10b981' }}>$24,774</div>
            </div>
            <div style={{
              background: 'rgba(59, 130, 246, 0.15)',
              border: '1px solid rgba(59, 130, 246, 0.4)',
              borderRadius: '12px',
              padding: '12px 20px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '11px', color: '#93c5fd', textTransform: 'uppercase', letterSpacing: '1px' }}>Best Threshold</div>
              <div style={{ fontSize: '24px', fontWeight: '700', color: '#3b82f6' }}>16.8%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '24px',
        flexWrap: 'wrap',
      }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 20px',
              borderRadius: '10px',
              border: activeTab === tab.id ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid rgba(71, 85, 105, 0.5)',
              background: activeTab === tab.id 
                ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))'
                : 'rgba(30, 41, 59, 0.5)',
              color: activeTab === tab.id ? '#93c5fd' : '#94a3b8',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'all 0.2s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '20px' }}>
          {/* Key Metrics */}
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>📊 System Performance</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              {[
                { label: 'Portfolio Value', value: '$1.45M', color: '#a78bfa' },
                { label: 'Best Model AUC', value: '0.6696', color: '#60a5fa' },
                { label: 'Approval Rate', value: '24.8%', color: '#34d399' },
                { label: 'Default Rate', value: '6.5%', color: '#f87171' },
              ].map((metric, i) => (
                <div key={i} style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  borderRadius: '12px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>{metric.label}</div>
                  <div style={{ fontSize: '22px', fontWeight: '700', color: metric.color }}>{metric.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Model Comparison */}
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>🤖 Model Performance</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={modelData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" domain={[0, 1]} stroke="#64748b" />
                <YAxis dataKey="model" type="category" width={120} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Bar dataKey="auc" name="AUC-ROC" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Quick Policy Selector */}
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
            gridColumn: 'span 2',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>⚡ Quick Policy Simulator</h3>
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '20px' }}>
              {policyData.map(p => (
                <button
                  key={p.policy}
                  onClick={() => setSelectedPolicy(p.policy)}
                  style={{
                    padding: '10px 16px',
                    borderRadius: '8px',
                    border: selectedPolicy === p.policy ? `2px solid ${p.color}` : '1px solid #475569',
                    background: selectedPolicy === p.policy ? `${p.color}20` : 'transparent',
                    color: selectedPolicy === p.policy ? p.color : '#94a3b8',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '500'
                  }}
                >
                  {p.policy}
                </button>
              ))}
            </div>
            {selectedPolicyData && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: '16px',
                background: 'rgba(15, 23, 42, 0.6)',
                borderRadius: '12px',
                padding: '20px'
              }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>THRESHOLD</div>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: selectedPolicyData.color }}>
                    {(selectedPolicyData.threshold * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>TOTAL PROFIT</div>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: selectedPolicyData.profit >= 0 ? '#10b981' : '#ef4444' }}>
                    {formatCurrency(selectedPolicyData.profit)}
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>APPROVAL RATE</div>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: '#60a5fa' }}>
                    {selectedPolicyData.approvalRate}%
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>DEFAULT RATE</div>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: '#f87171' }}>
                    {selectedPolicyData.defaultRate}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Threshold Optimization Tab */}
      {activeTab === 'threshold' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '20px' }}>
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>💰 Profit vs. Threshold</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={thresholdData}>
                <defs>
                  <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="threshold" tickFormatter={(v) => `${(v*100).toFixed(0)}%`} stroke="#64748b" />
                <YAxis tickFormatter={formatCurrency} stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  formatter={(value, name) => [formatCurrency(value), 'Total Profit']}
                  labelFormatter={(v) => `Threshold: ${(v*100).toFixed(1)}%`}
                />
                <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="5 5" />
                <ReferenceLine x={0.168} stroke="#3b82f6" strokeWidth={2} label={{ value: 'Optimal', fill: '#3b82f6', fontSize: 12 }} />
                <Area type="monotone" dataKey="profit" fill="url(#profitGradient)" stroke="#10b981" strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>📊 Approval vs. Default Rate</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={thresholdData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="threshold" tickFormatter={(v) => `${(v*100).toFixed(0)}%`} stroke="#64748b" />
                <YAxis tickFormatter={(v) => `${v}%`} stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelFormatter={(v) => `Threshold: ${(v*100).toFixed(1)}%`}
                />
                <Legend />
                <ReferenceLine x={0.168} stroke="#3b82f6" strokeDasharray="5 5" />
                <Line type="monotone" dataKey="approvalRate" name="Approval Rate %" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981' }} />
                <Line type="monotone" dataKey="defaultRate" name="Default Rate %" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Policy Comparison Tab */}
      {activeTab === 'policies' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '20px' }}>
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>💵 Total Profit by Policy</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={policyData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" tickFormatter={formatCurrency} stroke="#64748b" />
                <YAxis dataKey="policy" type="category" width={130} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  formatter={(value) => [formatCurrency(value), 'Profit']}
                />
                <ReferenceLine x={0} stroke="#ef4444" />
                <Bar dataKey="profit" radius={[0, 4, 4, 0]}>
                  {policyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.profit >= 0 ? '#10b981' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>⚖️ Risk-Return Trade-off</h3>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="defaultRate" name="Default Rate" tickFormatter={(v) => `${v}%`} stroke="#64748b" label={{ value: 'Default Rate %', position: 'bottom', fill: '#64748b' }} />
                <YAxis dataKey="profit" name="Profit" tickFormatter={formatCurrency} stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  formatter={(value, name) => [name === 'profit' ? formatCurrency(value) : `${value}%`, name]}
                  labelFormatter={(_, payload) => payload[0]?.payload?.policy || ''}
                />
                <Scatter data={policyData} shape="circle">
                  {policyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Policy Cards */}
          <div style={{ gridColumn: 'span 2' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '16px', color: '#f1f5f9' }}>📋 Policy Details</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
              {policyData.map(p => (
                <div key={p.policy} style={{
                  background: `linear-gradient(135deg, ${p.color}10, ${p.color}05)`,
                  border: `1px solid ${p.color}40`,
                  borderRadius: '12px',
                  padding: '16px',
                }}>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: p.color, marginBottom: '12px' }}>{p.policy}</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Threshold</span>
                      <span style={{ color: '#e2e8f0' }}>{(p.threshold * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Profit</span>
                      <span style={{ color: p.profit >= 0 ? '#10b981' : '#ef4444' }}>{formatCurrency(p.profit)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Approval</span>
                      <span style={{ color: '#e2e8f0' }}>{p.approvalRate}%</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Default</span>
                      <span style={{ color: '#e2e8f0' }}>{p.defaultRate}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Stress Testing Tab */}
      {activeTab === 'stress' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '20px' }}>
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>🔬 Stress Test Results</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={stressData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="scenario" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={formatCurrency} stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  formatter={(value) => [formatCurrency(value), 'Profit']}
                />
                <Bar dataKey="profit" radius={[4, 4, 0, 0]}>
                  {stressData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={['#10b981', '#84cc16', '#eab308', '#f97316', '#ef4444'][index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>📈 Interest Rate Sensitivity</h3>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={interestSensitivity}>
                <defs>
                  <linearGradient id="interestGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="rate" tickFormatter={(v) => `${v}%`} stroke="#64748b" label={{ value: 'Interest Rate', position: 'bottom', fill: '#64748b' }} />
                <YAxis tickFormatter={formatCurrency} stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  formatter={(value) => [formatCurrency(value), 'Profit']}
                  labelFormatter={(v) => `Interest Rate: ${v}%`}
                />
                <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="5 5" />
                <ReferenceLine x={8} stroke="#10b981" strokeWidth={2} label={{ value: 'Current', fill: '#10b981', fontSize: 12 }} />
                <Area type="monotone" dataKey="profit" stroke="#3b82f6" strokeWidth={2} fill="url(#interestGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Stress Scenario Cards */}
          <div style={{ gridColumn: 'span 2' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '16px', color: '#f1f5f9' }}>📊 Scenario Analysis</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px' }}>
              {stressData.map((s, i) => (
                <div key={s.scenario} style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  border: `1px solid ${['#10b981', '#84cc16', '#eab308', '#f97316', '#ef4444'][i]}40`,
                  borderRadius: '12px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '13px', fontWeight: '600', color: ['#10b981', '#84cc16', '#eab308', '#f97316', '#ef4444'][i] }}>
                    {s.scenario}
                  </div>
                  <div style={{ fontSize: '10px', color: '#64748b', margin: '4px 0' }}>
                    {s.multiplier}x Default Rate
                  </div>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: s.profit > 0 ? '#10b981' : '#ef4444', marginTop: '8px' }}>
                    {formatCurrency(s.profit)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Risk Drivers Tab */}
      {activeTab === 'features' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '20px' }}>
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>🎯 Top Risk Drivers</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={featureImportance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" domain={[0, 0.2]} tickFormatter={(v) => `${(v*100).toFixed(0)}%`} stroke="#64748b" />
                <YAxis dataKey="feature" type="category" width={140} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  formatter={(value) => [`${(value*100).toFixed(1)}%`, 'Importance']}
                />
                <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]}>
                  {featureImportance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={`hsl(${260 - index * 8}, 70%, ${60 - index * 2}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            borderRadius: '16px',
            padding: '24px',
          }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#f1f5f9' }}>📋 Key Risk Factors</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {[
                { factor: 'Checking Balance', desc: 'Negative balance indicates 15% higher default risk', risk: 'high' },
                { factor: 'Credit History', desc: 'Critical/poor history adds 25% to default probability', risk: 'high' },
                { factor: 'Loan Amount', desc: 'Amounts >$10K increase risk by 5%', risk: 'medium' },
                { factor: 'Employment Duration', desc: '>7 years tenure reduces risk by 10%', risk: 'low' },
                { factor: 'Age', desc: 'Ages 35-50 show lowest default rates', risk: 'low' },
              ].map((item, i) => (
                <div key={i} style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  borderRadius: '10px',
                  padding: '14px 16px',
                  borderLeft: `3px solid ${item.risk === 'high' ? '#ef4444' : item.risk === 'medium' ? '#f59e0b' : '#10b981'}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: '600', color: '#f1f5f9', fontSize: '14px' }}>{item.factor}</span>
                    <span style={{ 
                      fontSize: '10px', 
                      padding: '3px 8px', 
                      borderRadius: '4px',
                      background: item.risk === 'high' ? '#ef444420' : item.risk === 'medium' ? '#f59e0b20' : '#10b98120',
                      color: item.risk === 'high' ? '#ef4444' : item.risk === 'medium' ? '#f59e0b' : '#10b981',
                      textTransform: 'uppercase',
                      fontWeight: '600'
                    }}>
                      {item.risk} impact
                    </span>
                  </div>
                  <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '6px' }}>{item.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{
        marginTop: '24px',
        padding: '16px 24px',
        background: 'rgba(30, 41, 59, 0.4)',
        borderRadius: '12px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div style={{ fontSize: '12px', color: '#64748b' }}>
          Decision Optimization System • German Credit Dataset • 1,000 Applications
        </div>
        <div style={{ display: 'flex', gap: '16px', fontSize: '12px' }}>
          <span style={{ color: '#64748b' }}>Model: <span style={{ color: '#60a5fa' }}>Random Forest</span></span>
          <span style={{ color: '#64748b' }}>Interest: <span style={{ color: '#10b981' }}>8%</span></span>
          <span style={{ color: '#64748b' }}>Loss Rate: <span style={{ color: '#f87171' }}>75%</span></span>
        </div>
      </div>
    </div>
  );
}
