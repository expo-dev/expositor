pub fn format_time(t : f64) -> String
{
  // Formats time as
  //   ⟨·0.000⟩ when   60 > t >=    0
  //   ⟨·0'00"⟩ when 3600 > t >=   60
  //   ⟨·0:00'⟩ when        t >= 3600

  if t < 60.0 { return format!("{t:6.3}"); }

  let minutes = (t / 60.0).trunc();
  let seconds = t - minutes * 60.0;
  if t < 3600.0 {
    return format!("{minutes:2.0}'{seconds:02.0}\"");
  }

  let hours = (t / 3600.0).trunc();
  let minutes = (t - hours * 3600.0) / 60.0;
  return format!("{hours:2.0}:{minutes:02.0}'");
}

pub fn format_node_compact(n : usize) -> String
{
  // Formats nodes or nodes per second as
  //
  //   ⟨····0⟩  when            100 000 > n >=                  0
  //   ⟨··00k⟩  when         10 000 000 > n >=             10 000
  //   ⟨··00m⟩  when     10 000 000 000 > n >=         10 000 000
  //   ⟨··00g⟩  when 10 000 000 000 000 > n >=     10 000 000 000
  //   ⟨··00t⟩  when                      n >= 10 000 000 000 000

  if n <            100_000 {                                                    return format!("{n:5}" ); }
  if n <         10_000_000 { let r = (n +             499) /             1_000; return format!("{r:4}K"); }
  if n <     10_000_000_000 { let r = (n +         499_999) /         1_000_000; return format!("{r:4}M"); }
  if n < 10_000_000_000_000 { let r = (n +     499_999_999) /     1_000_000_000; return format!("{r:4}G"); }
                              let r = (n + 499_999_999_999) / 1_000_000_000_000; return format!("{r:4}T");
}

pub fn format_node(n : usize) -> String
{
  // Formats nodes or nodes per second as
  //
  //   ⟨···0  ⟩ when             10 000 > n >=                  0
  //   ⟨··00 k⟩ when         10 000 000 > n >=             10 000
  //   ⟨··00 m⟩ when     10 000 000 000 > n >=         10 000 000
  //   ⟨··00 g⟩ when 10 000 000 000 000 > n >=     10 000 000 000
  //   ⟨··00 t⟩ when                      n >= 10 000 000 000 000
  //
  // so that "12 node", "12 knode", "12 mnode/s", "12 mnps" can be written.

  if n <             10_000 {                                                    return format!("{n:4}  "); }
  if n <         10_000_000 { let r = (n +             499) /             1_000; return format!("{r:4} k"); }
  if n <     10_000_000_000 { let r = (n +         499_999) /         1_000_000; return format!("{r:4} m"); }
  if n < 10_000_000_000_000 { let r = (n +     499_999_999) /     1_000_000_000; return format!("{r:4} g"); }
                              let r = (n + 499_999_999_999) / 1_000_000_000_000; return format!("{r:4} t");
}
