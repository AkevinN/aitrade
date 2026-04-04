import React, { useState, useCallback } from 'react'
import {
  Card, Typography, Space, Button, Upload, Table, Tag, Alert,
  Select, Radio, Divider, Descriptions, Spin, App,
} from 'antd'
import { InboxOutlined, UploadOutlined } from '@ant-design/icons'
import type { UploadFile, UploadProps } from 'antd'
import { alphaService } from '../../../api/alpha'
import type { CsvPreviewResult, CsvImportMode, CsvInterval } from '../../../types/alpha'

const { Text } = Typography
const { Dragger } = Upload

interface CsvImportProps {
  onImportComplete?: () => void
}

const CsvImport: React.FC<CsvImportProps> = ({ onImportComplete }) => {
  const { message } = App.useApp()
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [previewResult, setPreviewResult] = useState<CsvPreviewResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [importing, setImporting] = useState(false)
  const [interval, setInterval] = useState<CsvInterval>('d')
  const [importMode, setImportMode] = useState<CsvImportMode>('merge')

  const handlePreview = useCallback(async (file: File) => {
    setLoading(true)
    setPreviewResult(null)
    try {
      const result = await alphaService.previewCsvImport(file)
      setPreviewResult(result)
      if (result.missing_required.length > 0) {
        message.warning(`Missing required fields: ${result.missing_required.join(', ')}`)
      } else {
        message.success('CSV parsed successfully. Review the mapping below.')
      }
    } catch (err) {
      message.error('Failed to parse CSV file')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleImport = useCallback(async () => {
    if (!previewResult || fileList.length === 0) {
      message.warning('Please upload a CSV file first')
      return
    }

    if (previewResult.missing_required.length > 0) {
      message.warning('Cannot import: missing required fields')
      return
    }

    setImporting(true)
    try {
      const file = fileList[0].originFileObj as File
      const result = await alphaService.importCsvData(file, interval, importMode)
      if (result.success) {
        message.success(result.message)
        setFileList([])
        setPreviewResult(null)
        onImportComplete?.()
      } else {
        message.error(result.message)
      }
    } catch (err) {
      message.error('Failed to import CSV')
      console.error(err)
    } finally {
      setImporting(false)
    }
  }, [previewResult, fileList, interval, importMode, onImportComplete])

  const uploadProps: UploadProps = {
    name: 'file',
    fileList,
    beforeUpload: (file) => {
      if (!file.name.toLowerCase().endsWith('.csv')) {
        message.error('Only CSV files are supported')
        return false
      }
      handlePreview(file)
      return false
    },
    onChange: ({ fileList: newFileList }) => {
      setFileList(newFileList)
    },
    onRemove: () => {
      setFileList([])
      setPreviewResult(null)
    },
    maxCount: 1,
    accept: '.csv',
  }

  const matchedColumns = previewResult ? Object.entries(previewResult.matched_fields) : []
  const standardFieldNames: Record<string, string> = {
    datetime: 'Datetime',
    symbol: 'Symbol',
    open: 'Open',
    high: 'High',
    low: 'Low',
    close: 'Close',
    volume: 'Volume',
    turnover: 'Turnover',
    open_interest: 'OI',
    change_pct: 'Chg%',
    amplitude: 'Amp%',
  }

  // Get mapped columns to display (only show columns that have data)
  const mappedColumns = previewResult?.columns || []

  return (
    <Card title="Import CSV Data" size="small" style={{ marginTop: 16 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* Upload Area */}
        <Dragger {...uploadProps} style={{ padding: '20px 0' }}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">Click or drag CSV file to upload</p>
          <p className="ant-upload-hint">
            Supports common field names in Chinese and English
          </p>
        </Dragger>

        {loading && (
          <div style={{ textAlign: 'center', padding: 20 }}>
            <Spin />
          </div>
        )}

        {/* Preview Result */}
        {previewResult && !loading && (
          <>
            <Divider style={{ margin: '12px 0' }} />

            {/* Summary Info */}
            <Descriptions size="small" column={2} bordered>
              <Descriptions.Item label="Total Rows">
                {previewResult.total_rows.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Symbols Found">
                {previewResult.symbols.length > 0
                  ? `${previewResult.symbols.length} (${previewResult.symbols.slice(0, 3).join(', ')}${previewResult.symbols.length > 3 ? '...' : ''})`
                  : 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Date Range">
                {previewResult.date_range[0] || 'N/A'} ~ {previewResult.date_range[1] || 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Columns">
                {previewResult.columns.length}
              </Descriptions.Item>
            </Descriptions>

            {/* Warning for missing fields */}
            {previewResult.missing_required.length > 0 && (
              <Alert
                type="warning"
                message="Missing Required Fields"
                description={`The following required fields are missing: ${previewResult.missing_required.join(', ')}. Import will fail.`}
                showIcon
              />
            )}

            {/* Unmapped columns warning */}
            {previewResult.unmapped_columns.length > 0 && (
              <Alert
                type="info"
                message="Unmapped Columns"
                description={`These columns were not recognized and will be ignored: ${previewResult.unmapped_columns.join(', ')}`}
                showIcon
              />
            )}

            {/* Field Mapping Table */}
            <div>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>
                Field Mapping (字段映射)
              </Text>
              <Table
                size="small"
                dataSource={matchedColumns.map(([std, csv], idx) => ({
                  key: idx,
                  standard: standardFieldNames[std] || std,
                  csvColumn: csv,
                }))}
                columns={[
                  { title: 'Standard Field', dataIndex: 'standard', key: 'standard', width: 150 },
                  {
                    title: 'CSV Column',
                    dataIndex: 'csvColumn',
                    key: 'csvColumn',
                    render: (v: string) => <Tag color="green">{v}</Tag>,
                  },
                ]}
                pagination={false}
                scroll={{ x: 300 }}
              />
            </div>

            {/* Sample Data Preview */}
            <div>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>
                Sample Data (前5行)
              </Text>
              <Table
                size="small"
                dataSource={previewResult.sample_rows.map((row, idx) => ({
                  ...row,
                  key: idx,
                }))}
                columns={mappedColumns.map((col) => ({
                  title: standardFieldNames[col] || col,
                  dataIndex: col,
                  key: col,
                  ellipsis: true,
                  width: 100,
                  render: (v: unknown) => {
                    if (v === null || v === undefined) return '-'
                    if (typeof v === 'number') return v.toLocaleString()
                    return String(v)
                  },
                }))}
                pagination={false}
                scroll={{ x: mappedColumns.length * 100 }}
              />
            </div>

            {/* Import Options */}
            <Divider style={{ margin: '12px 0' }} />

            <Space wrap>
              <div>
                <Text type="secondary" style={{ display: 'block', marginBottom: 4 }}>
                  Interval:
                </Text>
                <Radio.Group
                  value={interval}
                  onChange={(e) => setInterval(e.target.value)}
                  optionType="button"
                  buttonStyle="solid"
                >
                  <Radio.Button value="d">Daily</Radio.Button>
                  <Radio.Button value="m">Minute</Radio.Button>
                </Radio.Group>
              </div>

              <div>
                <Text type="secondary" style={{ display: 'block', marginBottom: 4 }}>
                  Import Mode:
                </Text>
                <Radio.Group
                  value={importMode}
                  onChange={(e) => setImportMode(e.target.value)}
                  optionType="button"
                  buttonStyle="solid"
                >
                  <Radio.Button value="merge">Merge (追加)</Radio.Button>
                  <Radio.Button value="replace">Replace (替换)</Radio.Button>
                </Radio.Group>
              </div>
            </Space>

            {/* Import Button */}
            <Button
              type="primary"
              icon={<UploadOutlined />}
              onClick={handleImport}
              loading={importing}
              disabled={previewResult.missing_required.length > 0}
              block
              style={{ marginTop: 8 }}
            >
              Import Data ({previewResult.total_rows.toLocaleString()} rows)
            </Button>
          </>
        )}
      </Space>
    </Card>
  )
}

export default CsvImport
