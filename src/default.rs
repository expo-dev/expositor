// use crate::nnue::Network;
use crate::nnext::FNetwork;
// use crate::policy::PolicyNetwork;

#[repr(C)]
#[repr(align(32))]
struct Container<T : ?Sized>(T);

const  VALUE_SIZE : usize = std::mem::size_of::<FNetwork>();
// const POLICY_SIZE : usize = std::mem::size_of::<PolicyNetwork>();

    static VALUE_ALIGNED   : &'static Container<[u8;  VALUE_SIZE]> = &Container(*include_bytes!("default.nnue"));
pub static DEFAULT_NETWORK : &'static           [u8;  VALUE_SIZE]  = &VALUE_ALIGNED.0;

//     static POLICY_ALIGNED  : &'static Container<[u8; POLICY_SIZE]> = &Container(*include_bytes!("default.kdnn"));
// pub static DEFAULT_POLICY  : &'static           [u8; POLICY_SIZE]  = &POLICY_ALIGNED.0;
