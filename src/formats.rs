pub fn format_time(t : f64) -> String
{
  // Formats time as
  //   ⟨·0.000⟩ when   60 > t >=    0
  //   ⟨·0'00"⟩ when 3600 > t >=   60
  //   ⟨·0:00'⟩ when        t >= 3600

  // TODO color

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
  //   ⟨··00K⟩  when         10 000 000 > n >=            100 000
  //   ⟨··00M⟩  when     10 000 000 000 > n >=         10 000 000
  //   ⟨··00G⟩  when 10 000 000 000 000 > n >=     10 000 000 000
  //   ⟨··00T⟩  when                      n >= 10 000 000 000 000

  // We can add color because this is only used
  //   in search, printing to standard error.
  // let u;
  // if n < 1000 { u = String::from(""); }
  // else {
  //   let mut z = ((n as f64).log10() - 5.0) / 5.0;
  //   if z > 1.0 { z = 1.0; }
  //   let l = 0.5 + 0.2 * z;
  //   let c = 0.12 + 0.06 * z;
  //   let h = 150.0 / 360.0;
  //   u = crate::term::Lch::new(l, c, h).to_rgb().fg();
  // }
  let q = "\x1B[38:5:242m";
  let u = "\x1B[38:5:238m";
  let d = "\x1B[39m";

  if n <            100_000 {                                                    return format!("{q}{n:5}{d}" ); }
  if n <         10_000_000 { let r = (n +             499) /             1_000; return format!("{q}{r:4}{u}ᵏ{d}"); }
  if n <     10_000_000_000 { let r = (n +         499_999) /         1_000_000; return format!("{q}{r:4}{u}ᵐ{d}"); }
  if n < 10_000_000_000_000 { let r = (n +     499_999_999) /     1_000_000_000; return format!("{q}{r:4}{u}ᵍ{d}"); }
                              let r = (n + 499_999_999_999) / 1_000_000_000_000; return format!("{q}{r:4}{u}ᵗ{d}");
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

pub fn variable_format_time(t : f64) -> String
{
  // Formats time as
  //   ⟨0.000⟩ when   60 > t >=    0
  //   ⟨0'00"⟩ when 3600 > t >=   60
  //   ⟨0:00'⟩ when        t >= 3600

  if t < 60.0 { return format!("{t:.3}"); }

  let minutes = (t / 60.0).trunc();
  let seconds = t - minutes * 60.0;
  if t < 3600.0 {
    return format!("{minutes:.0}'{seconds:02.0}\"");
  }

  let hours = (t / 3600.0).trunc();
  let minutes = (t - hours * 3600.0) / 60.0;
  return format!("{hours:.0}:{minutes:02.0}'");
}

pub fn variable_format_node(n : usize) -> String
{
  // Formats nodes or nodes per second as
  //
  //    ⟨0  ⟩ when            100 000 > n >=                  0
  //   ⟨00 k⟩ when         10 000 000 > n >=            100 000
  //   ⟨00 m⟩ when     10 000 000 000 > n >=         10 000 000
  //   ⟨00 g⟩ when 10 000 000 000 000 > n >=     10 000 000 000
  //   ⟨00 t⟩ when                      n >= 10 000 000 000 000
  //
  // so that "12 node", "12 knode", "12 mnode/s", "12 mnps" can be written.

  if n <            100_000 {                                                    return format!("{n} " ); }
  if n <         10_000_000 { let r = (n +             499) /             1_000; return format!("{r} k"); }
  if n <     10_000_000_000 { let r = (n +         499_999) /         1_000_000; return format!("{r} m"); }
  if n < 10_000_000_000_000 { let r = (n +     499_999_999) /     1_000_000_000; return format!("{r} g"); }
                              let r = (n + 499_999_999_999) / 1_000_000_000_000; return format!("{r} t");
}
