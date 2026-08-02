import net.minecraftforge.fml.common.FMLCommonHandler
import net.minecraftforge.fml.relauncher.Side

if (FMLCommonHandler.instance().getSide() != Side.CLIENT) {
    println "[HEAI] 服务端环境，跳过。"
    return
}

import mezz.jei.Internal
import net.minecraft.client.Minecraft
import net.minecraft.client.util.ITooltipFlag
import net.minecraft.item.Item
import net.minecraft.item.ItemStack
import net.minecraft.util.text.TextFormatting
import net.minecraft.creativetab.CreativeTabs
import net.minecraft.util.NonNullList
import net.minecraft.init.Items
import net.minecraftforge.fluids.FluidRegistry
import net.minecraftforge.fluids.FluidStack
import net.minecraftforge.fml.common.registry.ForgeRegistries
import java.net.URL
import java.net.HttpURLConnection
import java.io.OutputStreamWriter
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.atomic.AtomicBoolean

println "[HEAI] 确认为客户端环境，脚本开始初始化..."

def mc = Minecraft.getMinecraft()

def 配置 = [
    启用Tooltip采集: true
]

def 状态 = [
    上次搜索词: "",
    上次变化时间: 0L,
    索引已构建: false,
    正在检索: false,
    API: "http://127.0.0.1:27865",
    隐式注入活跃: false,
    原始搜索词: "",
    匹配关键词: ""
]

def 主循环运行中 = new AtomicBoolean(false)

def 搜索框 = null
def 搜索框原始Update = null

def 获取搜索框 = {
    if (搜索框 == null) {
        try {
            def runtime = Internal.getRuntime()
            if (runtime != null) {
                def overlay = runtime.getIngredientListOverlay()
                if (overlay != null) {
                    def field = overlay.getClass().getDeclaredField("searchField")
                    field.setAccessible(true)
                    搜索框 = field.get(overlay)
                    if (搜索框 != null) {
                        搜索框原始Update = 搜索框.metaClass.getMetaMethod('update')
                        println "[HEAI] 搜索框反射成功"
                    }
                }
            }
        } catch (Throwable t) {
            println "[HEAI] 搜索框反射失败: " + t.getMessage()
        }
    }
    搜索框
}

def 转义 = { String s -> s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"") }

def 发POST = { String url, String body ->
    try {
        def conn = (HttpURLConnection) new URL(url).openConnection()
        conn.setRequestMethod("POST")
        conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
        conn.setDoOutput(true)
        conn.setConnectTimeout(3000)
        conn.setReadTimeout(30000)
        def w = new OutputStreamWriter(conn.getOutputStream(), "UTF-8")
        w.write(body); w.flush(); w.close()
        def code = conn.getResponseCode()
        def is = code >= 400 ? conn.getErrorStream() : conn.getInputStream()
        def r = new BufferedReader(new InputStreamReader(is, "UTF-8"))
        def sb = new StringBuilder()
        String line
        while ((line = r.readLine()) != null) sb.append(line)
        r.close()
        return sb.toString()
    } catch (Throwable t) { return "" }
}

def 含特殊字符 = { String s ->
    s == null || s ==~ /.*[!@#\$%^&*()+\-=\[\]{};':"\\|,.<>\/?~`].*/
}

def 解析物品 = { String res ->
    def 物品 = []
    try {
        def resultIdx = res.indexOf('"result"')
        if (resultIdx >= 0) {
            def bs = res.indexOf('[[', resultIdx)
            def be = res.lastIndexOf(']]')
            if (bs >= 0 && be > bs) {
                def inner = res.substring(bs + 2, be)
                def parts = inner.split(/\]\s*,\s*\[/)
                for (def p : parts) {
                    def cleaned = p.replaceAll(/^\[/, '').replaceAll(/\]$/, '')
                    def m = (cleaned =~ /"([^"]*)"/)
                    while (m.find()) {
                        def item = m.group(1)
                        if (item && item.trim()) 物品.add(item.trim())
                    }
                }
            }
        }
        if (物品.isEmpty()) {
            def m = (res =~ /"([^"]+)"/)
            while (m.find()) {
                def v = m.group(1)
                if (v != "status" && v != "ok" && v != "result" && v != "message") 物品.add(v)
            }
        }
    } catch (Throwable e) {
        println "[HEAI] JSON解析: " + e.getMessage()
    }
    物品
}

def 轮询线程 = new Thread({
    Thread.sleep(5000)
    println "[HEAI] 后台轮询启动！"

    while (true) {
        try {
            Thread.sleep(250)

            mc.addScheduledTask({
                if (主循环运行中.get()) return
                主循环运行中.set(true)

                try {
                    def runtime = Internal.getRuntime()
                    if (runtime == null) return
                    def filter = runtime.getIngredientFilter()
                    if (filter == null) return

                    if (!状态.索引已构建) {
                        状态.索引已构建 = true
                        println "[HEAI] 收集物品..."
                        def 条目列表 = []
                        def reg = null
                        try { reg = ForgeRegistries.ITEMS } catch (Throwable t) {}
                        if (reg == null) try { reg = Item.REGISTRY } catch (Throwable t) {}
                        if (reg != null) {
                            for (def item : reg) {
                                if (item == null) continue
                                def list = NonNullList.create()
                                try { item.getSubItems(CreativeTabs.SEARCH, list) } catch (Throwable t) {}
                                if (list.isEmpty()) list.add(new ItemStack(item))
                                for (def stack : list) {
                                    if (stack == null || stack.isEmpty() || stack.getItem() == Items.AIR) continue
                                    try {
                                        def name = TextFormatting.getTextWithoutFormattingCodes(stack.getDisplayName())
                                        if (!name || name.trim().isEmpty()) continue
                                        if (配置.启用Tooltip采集) {
                                            def 描述 = []
                                            def tooltipLines = stack.getTooltip(null, ITooltipFlag.TooltipFlags.NORMAL)
                                            if (tooltipLines) {
                                                for (def line : tooltipLines) {
                                                    if (line.contains("§8")) continue
                                                    def clean = TextFormatting.getTextWithoutFormattingCodes(line)
                                                    if (clean && !clean.trim().isEmpty() && clean.length() > 1) {
                                                        描述.add(clean.toString())
                                                    }
                                                }
                                            }
                                            条目列表.add([name.toString(), 描述.join(", ")])
                                        } else {
                                            条目列表.add([name.toString()])
                                        }
                                    } catch (Throwable t) {}
                                }
                            }
                        }
                        def 流体数量 = 0
                        try {
                            for (def fluid : FluidRegistry.getRegisteredFluids().values()) {
                                if (fluid == null) continue
                                try {
                                    def fluidStack = new FluidStack(fluid, 1000)
                                    def name = TextFormatting.getTextWithoutFormattingCodes(fluid.getLocalizedName(fluidStack))
                                    if (name && !name.trim().isEmpty()) {
                                        条目列表.add([name.toString()])
                                        流体数量++
                                    }
                                } catch (Throwable t) {}
                            }
                        } catch (Throwable t) {}
                        println "[HEAI] 收集到 " + 条目列表.size() + " 个物品（含 " + 流体数量 + " 个流体），后台上传中..."
                        def sb = new StringBuilder('{"data": [')
                        for (int i = 0; i < 条目列表.size(); i++) {
                            def 条目 = 条目列表.get(i)
                            if (条目.size() == 1) {
                                sb.append('"').append(转义(条目[0])).append('"')
                            } else {
                                sb.append('["').append(转义(条目[0])).append('", "').append(转义(条目[1])).append('"]')
                            }
                            if (i < 条目列表.size() - 1) sb.append(',')
                        }
                        sb.append(']}')
                        def body = sb.toString()
                        def api = 状态.API
                        new Thread({ 发POST(api + "/add", body); println "[HEAI] 索引构建完成！" } as Runnable).start()
                    }

                    def 当前 = filter.getFilterText() ?: ""

                    if (当前 != 状态.上次搜索词) {
                        状态.上次搜索词 = 当前
                        状态.上次变化时间 = System.currentTimeMillis()
                    }

                    if (状态.隐式注入活跃) {
                        def cfgNow = mezz.jei.config.Config.getFilterText()
                        def expected = 状态.原始搜索词 + " | " + 状态.匹配关键词
                        if (cfgNow != expected) {
                            状态.隐式注入活跃 = false
                            if (搜索框 != null && 搜索框原始Update != null) {
                                try {
                                    搜索框.metaClass.update = 搜索框原始Update
                                    println "[HEAI] 隐式注入结束"
                                } catch (Throwable t) {}
                            }
                        }
                    }

                    if (!状态.正在检索 && 当前
                        && !状态.隐式注入活跃
                        && !含特殊字符(当前)
                        && (System.currentTimeMillis() - 状态.上次变化时间) > 1200) {

                        状态.正在检索 = true
                        def 注入词 = 当前
                        def searchBody = '{"data": ["' + 转义(当前) + '"]}'
                        def api = 状态.API

                        new Thread({
                            try {
                                def res = 发POST(api + "/search", searchBody)
                                if (res.contains('"ok"')) {
                                    def 物品 = 解析物品(res)

                                    if (物品) {
                                        def 合并词 = 注入词 + " | " + 物品.take(5).join(" | ")
                                        def 匹配词 = 物品.take(5).join(" | ")

                                        mc.addScheduledTask({
                                            try {
                                                获取搜索框()

                                                if (搜索框 != null && 搜索框原始Update != null) {
                                                    搜索框.metaClass.update = { ->
                                                        try {
                                                            def igf = Internal.getIngredientFilter()
                                                            if (igf) {
                                                                def list = igf.getIngredientList()
                                                                delegate.setTextColor(list.size() == 0 ? 0xFF5555 : 0xFFFFFF)
                                                            }
                                                        } catch (Throwable t) {}
                                                    }
                                                }

                                                mezz.jei.config.Config.setFilterText(合并词)
                                                filter.invalidateCache()
                                                filter.notifyListenersOfChange()

                                                if (搜索框 != null) {
                                                    搜索框.setText(注入词)
                                                }

                                                状态.隐式注入活跃 = true
                                                状态.原始搜索词 = 注入词
                                                状态.匹配关键词 = 匹配词
                                                状态.上次搜索词 = 合并词
                                                状态.上次变化时间 = System.currentTimeMillis() + 999999

                                                println "[HEAI] 注入: 框='" + 注入词 + "' 搜='" + 合并词 + "'"
                                            } catch (Throwable t) {
                                                println "[HEAI] 注入失败: " + t.getMessage()
                                            } finally {
                                                状态.正在检索 = false
                                            }
                                        } as Runnable)
                                        return
                                    }
                                }
                            } catch (Throwable t) {
                                println "[HEAI] 检索异常: " + t.getMessage()
                            }
                            状态.正在检索 = false
                        } as Runnable).start()
                    }
                } catch (Throwable t) {
                    println "[HEAI] 主循环异常: " + t.getMessage()
                } finally {
                    主循环运行中.set(false)
                }
            } as Runnable)

        } catch (Throwable t) {
            println "[HEAI] 轮询异常: " + t.getMessage()
        }
    }
} as Runnable)

轮询线程.setDaemon(true)
轮询线程.start()

println "[HEAI] 已启动"
