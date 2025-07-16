local utils = require 'mp.utils'

local files = {}
local cur_index = 1

-- -- User interactions --

local function ask_zenity(question)
    local cmd = {"zenity", "--question", "--text", question}
    local res = mp.command_native({
        name = "subprocess",
        args = cmd,
        playback_only = false,
        capture_stdout = true,
    })
    if res.status == 0 then
        return true
    else
        return false
    end
end

local function notify_once(msg, secs)
    secs = secs or 5
    mp.osd_message(msg, secs)
    mp.msg.info(msg)
end

local function split_filename(fname)
    local base, ext = fname:match("^(.-)(%.[^%.]*)$")
    if not base then return fname, "" end
    return base, ext
end

local function is_main_alternative(filepath)
    local _, fname = utils.split_path(filepath)
    return fname:match("%+00_")
end

local function get_main_base(filepath)
    local _, fname = utils.split_path(filepath)
    local base = fname:match("^(.-)(%.[^%.]*)$") or fname
    base = base:gsub("%+%d+.*$", "")
    return base
end

-- Helper function to load file at time t
local function load_at(path, t)
    -- t in seconds (float); mpv will parse fractional seconds
    mp.commandv("loadfile", path, "replace", "-1", ("start=%f"):format(t))
end

-- -- File scanning and state setup --

local function get_files()
    local path = mp.get_property('path')
    if not path then return end

    local dir, filename = utils.split_path(path)
    local abs_dir = mp.command_native({"expand-path", dir})
    local scan_dir = utils.readdir(abs_dir, "files")
    if not scan_dir then return end

    -- Always compare using the main_base (removes +NN and extension)
    local current_main_base = get_main_base(filename)

    local found_main = nil
    local found_alts = {}

    for _, f in ipairs(scan_dir) do
        local f_main_base = get_main_base(f)
        if f_main_base == current_main_base then
            -- Is it a main file (no +NN) or alternative (+NN)?
            if f:find("%+%d+") then
                table.insert(found_alts, f)
            else
                found_main = f
            end
        end
    end

    -- Sort alternatives by their number
    table.sort(found_alts, function(a, b)
        local na = tonumber(a:match("%+(%d+)_"))
        local nb = tonumber(b:match("%+(%d+)_"))
        return (na or 999) < (nb or 999)
    end)

    files = {}
    if found_main then table.insert(files, utils.join_path(abs_dir, found_main)) end
    for _, f in ipairs(found_alts) do
        table.insert(files, utils.join_path(abs_dir, f))
    end

    cur_index = 1
    local current_abs = utils.join_path(abs_dir, filename)
    for i, f in ipairs(files) do
        if f == current_abs then
            cur_index = i
            break
        end
    end

    show_on_load(filename)
end

function show_menu_message(current_file)
    if #files <= 1 then
        notify_once(("Current: %s\nNo alternatives found."):format(current_file))
        return
    end

    local msg = ("Current: %s\nAlternatives:\n"):format(current_file)
    local user_alt_n = 1
    for i = 2, #files do
        local alt_name = files[i]
        if not is_main_alternative(alt_name) then
            local _, fname = utils.split_path(alt_name)
            local alt_index, alt_suffix = fname:match("%+(%d+)_([^%.]+)")
            if not alt_index then
                alt_index, alt_suffix = "??", fname
            end
            msg = msg .. string.format("%02d: %s\n", user_alt_n, alt_suffix or "?")
            user_alt_n = user_alt_n + 1
        end
    end
    msg = msg .. "\n[1-9]: Go to alternative\nTAB: Toggle main/+00\n</>: Prev/Next alternative"
    notify_once(msg, 5)
end

function show_on_load(current_file)
    if cur_index == 1 then
        show_menu_message(current_file)
    else
        notify_once(current_file, 2)
    end
end

-- -- Alternatives navigation --

local function get_user_alt_indices()
    local indices = {}
    for i = 2, #files do
        if not is_main_alternative(files[i]) then
            table.insert(indices, i)
        end
    end
    return indices
end

local function goto_alternative(n)
    if #files == 0 then return end
    local alt_indices = get_user_alt_indices()
    local alt_idx = alt_indices[n]
    if alt_idx then
        mp.commandv('loadfile', files[alt_idx], 'replace')
    end
end

local function toggle_tab()
    if #files == 0 then return end
    local cur_time = mp.get_property_number("time-pos", 0)
    if cur_index == 1 then
        for i, f in ipairs(files) do
            if is_main_alternative(f) then
                -- You can use start= as well if you want to preserve time when going to +00 alternative
                load_at(f, cur_time)
                return
            end
        end
    else
        load_at(files[1], cur_time)
    end
end

local function next_alt()
    if #files == 0 then return end
    local idx = cur_index
    while idx < #files do
        idx = idx + 1
        if not is_main_alternative(files[idx]) then
            mp.commandv('loadfile', files[idx], 'replace')
            return
        end
    end
end

local function prev_alt()
    if #files == 0 then return end
    local idx = cur_index
    while idx > 1 do
        idx = idx - 1
        if not is_main_alternative(files[idx]) then
            mp.commandv('loadfile', files[idx], 'replace')
            return
        end
    end
end

-- -- Main file navigation (next/prev main video) --

local function get_main_files(dir)
    local scan_dir = utils.readdir(dir, "files")
    if not scan_dir then return {} end
    local main_files = {}
    for _, f in ipairs(scan_dir) do
        if not f:find("+") then
            local full = utils.join_path(dir, f)
            local stat = utils.file_info(full)
            if stat and stat.is_file and not f:match("^%.") then
                table.insert(main_files, f)
            end
        end
    end
    table.sort(main_files)
    return main_files
end

local function jump_main_file(dir, filename, step)
    local abs_dir = mp.command_native({"expand-path", dir})
    local main_files = get_main_files(abs_dir)
    if #main_files == 0 then
        notify_once("No main files found in directory.", 3)
        return
    end
    -- Always use base name for matching
    local main_base = get_main_base(filename)
    local idx = nil
    for i, f in ipairs(main_files) do
        if get_main_base(f) == main_base then
            idx = i
            break
        end
    end
    if not idx then
        notify_once("Cannot locate matching main file for navigation.", 3)
        return
    end
    local new_idx = idx + step
    if new_idx < 1 or new_idx > #main_files then
        notify_once("No next/previous main file.", 3)
        return
    end
    local new_file = main_files[new_idx]
    mp.commandv('loadfile', utils.join_path(abs_dir, new_file), 'replace')
end

-- -- Save and stabilization --

local function save_clip()
    local path = mp.get_property('path')
    if not path then
        mp.osd_message("No file loaded.", 2)
        return
    end
    local dir, filename = utils.split_path(path)
    local keepdir = utils.join_path(dir, "keep")
    local stabilize = ask_zenity("Stabilize this clip before saving?")
    if stabilize == nil then
        mp.osd_message("Cancelled.", 2)
        return
    end
    if not utils.file_info(keepdir) then
        os.execute('mkdir -p "' .. keepdir .. '"')
    end
    local dest = utils.join_path(keepdir, filename)
    if stabilize then
        mp.command_native({
            name = "subprocess",
            args = {"stabilize", "-o", dest, path},
            detach = true,
            playback_only = false,
        })
        mp.osd_message("Stabilizing in background:\n" .. dest, 3)
    else
        mp.osd_message("Copying...", 2)
        mp.command_native_async({
            name = "subprocess",
            args = {"cp", path, dest},
            playback_only = false,
        }, function(result)
            if result == true or (type(result) == "table" and result.status == 0) then
                mp.osd_message("Saved to " .. dest, 2)
            elseif type(result) == "table" then
                mp.osd_message("Copy failed.", 3)
            else
                mp.osd_message("Copy failed to start.", 3)
            end
        end)
    end
end

-- -- Keybindings and event hooks --

mp.register_event("file-loaded", get_files)
for i = 1, 9 do
    mp.add_key_binding(tostring(i), "goto_alt_"..i, function() goto_alternative(i) end)
end
mp.add_key_binding("TAB", "toggle_tab", toggle_tab)
mp.add_key_binding(">", "next_alt", next_alt)
mp.add_key_binding("<", "prev_alt", prev_alt)
mp.register_event("start-file", get_files)

mp.add_key_binding("Ctrl+>", "next_main_file", function()
    local path = mp.get_property('path')
    if not path then return end
    local dir, filename = utils.split_path(path)
    jump_main_file(dir, filename, 1)
end)
mp.add_key_binding("Ctrl+<", "prev_main_file", function()
    local path = mp.get_property('path')
    if not path then return end
    local dir, filename = utils.split_path(path)
    jump_main_file(dir, filename, -1)
end)

mp.add_key_binding("s", "save_clip", save_clip)

-- -- Video/Audio reset and time sync --

local initial_mute = mp.get_property_native("mute")

local function reset_mute()
    if initial_mute ~= nil then
        mp.set_property_native("mute", initial_mute)
    end
end

local function on_file_load()
    get_files()
    mp.set_property("loop-file", "no")
    mp.set_property("video-zoom", "0")
    mp.set_property("video-pan-x", "0")
    mp.set_property("video-pan-y", "0")
    reset_mute()
end

mp.register_event("file-loaded", on_file_load)
