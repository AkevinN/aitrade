import React, { useState } from 'react'
import {
  Modal, Table, Tag, Typography, Space, Button, Statistic, Row, Col,
  Card, Descriptions, Alert, Divider, Tooltip,
} from 'antd'
import {
  EyeOutlined, CloudDownloadOutlined, UploadOutlined,
  DatabaseOutlined, FileTextOutlined, DeleteOutlined,
} from '@ant-design/icons'
import type { BarDataItem, BarDataDetail, BarDataItemWithInterval } from '../../../types/alpha'
import { alphaService } from '../../../api/alpha'

const { Text, Title } = Typography

interface DataViewerProps {
  record: BarDataItemWithInterval
  onClose: () => void
}

const DataViewer: React.FC<DataViewerProps> = ({ record, onClose }) => {
  const [detail, setDetail] = useState<BarDataDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const loadDetail = async () => {
    if (detail) return
    setLoading(true)
    setLoadError(null)
    try {
      // Use intervalType from record instead of guessing from symbol
      const data = await alphaService.getBarDataDetail(
        record.intervalType,
        record.vt_symbol,
        { limit: 100 }
      )
      setDetail(data)
    } catch (err: unknown) {
      const error = err as { response?: { status?: number } }
      // 404 means data file doesn't exist - show basic info only
      if (error?.response?.status === 404) {
        setDetail({
          vt_symbol: record.vt_symbol,
          interval: record.intervalType,
          row_count: record.row_count || 0,
          start: record.start || '',
          end: record.end || '',
          columns: [],
          preview: [],
          loaded_count: 0,
          has_more: false,
          next_before: null,
        })
      } else {
        setLoadError('Failed to load data details')
        console.error(err)
      }
    } finally {
      setLoading(false)
    }
  }

  React.useEffect(() => {
    // Reset state when record changes
    setDetail(null)
    setLoadError(null)
    if (record?.row_count !== undefined && record.row_count > 0) {
      loadDetail()
    }
  }, [record?.vt_symbol, record?.intervalType])

  const columns = detail?.columns.map((col) => ({
    title: col,
    dataIndex: col,
    key: col,
    width: 120,
    ellipsis: true,
    render: (v: unknown) => {
      if (v === null || v === undefined) return '-'
      if (typeof v === 'number') return v.toLocaleString(undefined, { maximumFractionDigits: 4 })
      return String(v)
    },
  })) || []

  const sourceIcon = record._source === 'csv' ? (
    <Tag color="purple" icon={<UploadOutlined />}>CSV Import</Tag>
  ) : (
    <Tag color="blue" icon={<CloudDownloadOutlined />}>Downloaded</Tag>
  )

  return (
    <Modal
      open={!!record}
      onCancel={onClose}
      footer={null}
      width={1000}
      title={
        <Space>
          <DatabaseOutlined />
          <span>Data Detail: {record?.vt_symbol}</span>
          {sourceIcon}
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* Data Summary */}
        <Descriptions size="small" column={4} bordered>
          <Descriptions.Item label="Total Rows">
            <Text strong>{record?.row_count?.toLocaleString()}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Date Range">
            <Text>{record?.start?.slice(0, 10)} ~ {record?.end?.slice(0, 10)}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="File Size">
            <Text>{record?.file_size_kb?.toFixed(1)} KB</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Preview Rows">
            <Text>{detail?.loaded_count || 0}</Text>
          </Descriptions.Item>
        </Descriptions>

        {/* Columns Info */}
        {detail?.columns && (
          <div>
            <Text type="secondary" style={{ marginBottom: 8, display: 'block' }}>
              <FileTextOutlined /> Columns ({detail.columns.length}):
            </Text>
            <Space wrap size={[4, 4]}>
              {detail.columns.map((col) => (
                <Tag key={col} style={{ margin: 0 }}>{col}</Tag>
              ))}
            </Space>
          </div>
        )}

        <Divider style={{ margin: '12px 0' }} />

        {/* Data Preview Table */}
        <div>
          <Text strong style={{ marginBottom: 8, display: 'block' }}>
            Data Preview (Latest 100 rows)
          </Text>
          {loadError && <Alert type="error" message={loadError} />}
          {detail?.preview && detail.preview.length > 0 ? (
            <Table
              size="small"
              columns={
                detail.columns.map((col) => ({
                  title: col,
                  dataIndex: col,
                  key: col,
                  width: 120,
                  ellipsis: true,
                  render: (v: unknown) => {
                    if (v === null || v === undefined) return '-'
                    if (typeof v === 'number') return v.toLocaleString(undefined, { maximumFractionDigits: 4 })
                    return String(v)
                  },
                }))
              }
              dataSource={detail.preview.map((row, idx) => ({ ...row, key: idx }))}
              loading={loading}
              pagination={detail.row_count > 20 ? { pageSize: 20 } : false}
              scroll={{ x: true, y: 400 }}
            />
          ) : (
            <Alert type="info" message="No preview data available" />
          )}
        </div>

        {detail?.has_more && (
          <Alert
            type="info"
            message={`Only showing latest ${detail.loaded_count} of ${detail.row_count.toLocaleString()} rows. Use pagination for more data.`}
          />
        )}
      </Space>
    </Modal>
  )
}

interface DataStatsProps {
  barData: { daily: BarDataItem[]; minute: BarDataItem[] } | undefined
  totalSymbols: number
  totalRows: number
}

export const DataStats: React.FC<DataStatsProps> = ({ barData, totalSymbols, totalRows }) => (
  <Card size="small" style={{ marginBottom: 16 }}>
    <Row gutter={16}>
      <Col span={6}>
        <Statistic
          title="Total Symbols"
          value={totalSymbols}
          prefix={<DatabaseOutlined />}
          valueStyle={{ fontSize: 18 }}
        />
      </Col>
      <Col span={6}>
        <Statistic
          title="Daily Count"
          value={barData?.daily?.length || 0}
          suffix="files"
          valueStyle={{ fontSize: 18 }}
        />
      </Col>
      <Col span={6}>
        <Statistic
          title="Minute Count"
          value={barData?.minute?.length || 0}
          suffix="files"
          valueStyle={{ fontSize: 18 }}
        />
      </Col>
      <Col span={6}>
        <Statistic
          title="Total Rows"
          value={totalRows}
          valueStyle={{ fontSize: 18 }}
        />
      </Col>
    </Row>
  </Card>
)

export interface BarDataItemWithInterval extends BarDataItem {
  intervalType: 'daily' | 'minute'
}

export interface DataListViewProps {
  barData: { daily: BarDataItem[]; minute: BarDataItem[] } | undefined
  onDelete: (intervalType: string, vtSymbol: string) => void
  onView: (record: BarDataItemWithInterval) => void
  loading?: boolean
}

export const DataListView: React.FC<DataListViewProps> = ({
  barData,
  onDelete,
  onView,
  loading,
}) => {
  const allData: BarDataItemWithInterval[] = [
    ...(barData?.daily?.map((d) => ({ ...d, intervalType: 'daily' as const, key: `daily_${d.vt_symbol}` })) || []),
    ...(barData?.minute?.map((d) => ({ ...d, intervalType: 'minute' as const, key: `minute_${d.vt_symbol}` })) || []),
  ]

  const columns = [
    {
      title: 'Symbol',
      dataIndex: 'vt_symbol',
      key: 'vt_symbol',
      width: 160,
      render: (v: string, record: BarDataItemWithInterval) => (
        <Space>
          <Tooltip title="Click to view data">
            <Button type="link" size="small" onClick={() => onView(record)} style={{ padding: 0 }}>
              {v}
            </Button>
          </Tooltip>
          <EyeOutlined
            style={{ cursor: 'pointer', color: '#1890ff' }}
            onClick={() => onView(record)}
          />
        </Space>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'intervalType',
      key: 'intervalType',
      width: 80,
      render: (v: string) => v === 'daily' ? <Tag color="green">Daily</Tag> : <Tag color="blue">Minute</Tag>,
    },
    {
      title: 'Rows',
      dataIndex: 'row_count',
      key: 'row_count',
      width: 100,
      render: (v: number) => v?.toLocaleString() ?? '-',
      sorter: (a: BarDataItem, b: BarDataItem) => (a.row_count || 0) - (b.row_count || 0),
    },
    {
      title: 'Date Range',
      dataIndex: 'start',
      key: 'range',
      width: 180,
      render: (_: unknown, record: BarDataItem) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {record.start?.slice(0, 10) || '-'} ~ {record.end?.slice(0, 10) || '-'}
        </Text>
      ),
    },
    {
      title: 'Size',
      dataIndex: 'file_size_kb',
      key: 'file_size_kb',
      width: 80,
      render: (v: number) => v ? `${v.toFixed(1)} KB` : '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 160,
      render: (_: unknown, record: BarDataItem & { intervalType: string }) => (
        <Space size="small">
          <Button
            size="small"
            type="primary"
            icon={<EyeOutlined />}
            onClick={() => onView(record)}
          >
            View
          </Button>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => onDelete(record.intervalType, record.vt_symbol)}
          />
        </Space>
      ),
    },
  ]

  return (
    <Table
      size="small"
      columns={columns}
      dataSource={allData}
      loading={loading}
      pagination={{
        pageSize: 10,
        showSizeChanger: true,
        showTotal: (total) => `Total ${total} files`,
      }}
      scroll={{ x: 700 }}
    />
  )
}

export default DataViewer
