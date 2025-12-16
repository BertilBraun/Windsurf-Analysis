-- mark.lua (place in same directory as the python script)
local mp = require 'mp'
local function start()
    mp.commandv('script-message', 'm_mark_start')
end
local function stop()
    mp.commandv('script-message', 'm_mark_end')
end

local function message_keep()
    mp.commandv('script-message', 'm_keep')
end

local function message_quit()
    mp.commandv('script-message', 'm_quit')
    mp.commandv('quit')
end

local function toggle_stabilize()
    mp.commandv('script-message', 'm_toggle_stabilize')
end

-- Don't name the keybindings the same as the messages otherwise they will be
-- automatically called
mp.add_key_binding('s', 'set_mark_start', start)
mp.add_key_binding('e', 'set_mark_end', stop)
mp.add_key_binding('t', 'mark_total_video', message_keep)
mp.add_key_binding('q', 'quit', message_quit)
mp.add_key_binding('S', 'toggle_stabilization', toggle_stabilize)
