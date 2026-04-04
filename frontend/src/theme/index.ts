import { theme } from 'antd'
import type { ThemeConfig } from 'antd'

const darkTheme: ThemeConfig = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#1668dc',
    colorInfo: '#1668dc',
    colorSuccess: '#49aa19',
    colorWarning: '#d89614',
    colorError: '#dc4446',
    colorBgBase: '#141414',
    colorBgContainer: '#1f1f1f',
    colorBgElevated: '#262626',
    colorBgLayout: '#0a0a0a',
    colorTextBase: '#e8e8e8',
    colorBorder: '#424242',
    colorBorderSecondary: '#303030',
    borderRadius: 6,
    fontFamily: "'Inter', 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif",
    fontSize: 13,
  },
  components: {
    Layout: {
      headerBg: '#141414',
      siderBg: '#1a1a1a',
      bodyBg: '#0a0a0a',
      headerHeight: 48,
    },
    Menu: {
      darkItemBg: '#1a1a1a',
      darkSubMenuItemBg: '#141414',
      darkItemSelectedBg: '#1668dc20',
    },
    Table: {
      headerBg: '#1a1a1a',
      rowHoverBg: '#262626',
      borderColor: '#303030',
    },
    Card: {
      colorBgContainer: '#1f1f1f',
    },
    Modal: {
      contentBg: '#1f1f1f',
      headerBg: '#1f1f1f',
    },
  },
}

export default darkTheme
